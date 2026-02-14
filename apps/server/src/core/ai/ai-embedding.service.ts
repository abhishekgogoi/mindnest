import { Injectable, Logger } from '@nestjs/common';
import { InjectKysely } from 'nestjs-kysely';
import { KyselyDB } from '@docmost/db/types/kysely.types';
import { EnvironmentService } from '../../integrations/environment/environment.service';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { embedMany, EmbeddingModel } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { toSql } from 'pgvector';
import { sql } from 'kysely';

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;
const EMBEDDING_BATCH_SIZE = 100;

interface PageForEmbedding {
  id: string;
  spaceId: string;
  workspaceId: string;
  textContent: string | null;
}

@Injectable()
export class EmbeddingService {
  private readonly logger = new Logger(EmbeddingService.name);

  private readonly textSplitter: RecursiveCharacterTextSplitter;

  constructor(
    @InjectKysely() private readonly db: KyselyDB,
    private readonly environmentService: EnvironmentService,
  ) {
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP,
    });
  }

  /**
   * Creates the embedding model instance based on the configured AI driver.
   */
  getEmbeddingModel(): EmbeddingModel {
    const driver = this.environmentService.getAiDriver();
    const modelName = this.environmentService.getAiEmbeddingModel();

    switch (driver) {
      case 'openai': {
        const openai = createOpenAI({
          apiKey: this.environmentService.getOpenAiApiKey(),
        });
        return openai.embedding(modelName);
      }
      case 'gemini': {
        const google = createGoogleGenerativeAI({
          apiKey: this.environmentService.getGeminiApiKey(),
        });
        return google.embedding(modelName);
      }
      case 'ollama': {
        const ollama = createOpenAICompatible({
          name: 'ollama',
          baseURL: this.environmentService.getOllamaApiUrl() + '/v1',
        });
        return ollama.embeddingModel(modelName);
      }
      default:
        throw new Error(`Unsupported AI driver: ${driver}`);
    }
  }

  /**
   * Resolves the embedding dimension from the environment config,
   * or uses the model's default dimension based on known presets.
   */
  private getEmbeddingDimension(): number {
    const envDimension = this.environmentService.getAiEmbeddingDimension();
    if (envDimension && !isNaN(envDimension)) {
      return envDimension;
    }

    // Known preset dimensions for common models
    const presetDimensions: Record<string, number> = {
      'text-embedding-3-small': 1536,
      'text-embedding-3-large': 3072,
      'text-embedding-ada-002': 1536,
      'text-embedding-004': 768,
    };

    const modelName = this.environmentService.getAiEmbeddingModel();
    return presetDimensions[modelName] ?? 1536;
  }

  /**
   * Generates embeddings for all pages in a workspace.
   */
  async generateEmbeddingsForWorkspace(workspaceId: string): Promise<void> {
    this.logger.log(`Starting embedding generation for workspace: ${workspaceId}`);

    const pages = await this.db
      .selectFrom('pages')
      .select(['id', 'spaceId', 'workspaceId', 'textContent'])
      .where('workspaceId', '=', workspaceId)
      .where('deletedAt', 'is', null)
      .execute();

    this.logger.log(`Found ${pages.length} pages to process`);

    let processedCount = 0;
    let skippedCount = 0;

    for (const page of pages) {
      if (!page.textContent || page.textContent.trim().length === 0) {
        skippedCount++;
        continue;
      }

      try {
        await this.generateEmbeddingsForPage(page);
        processedCount++;
      } catch (err) {
        this.logger.error(
          `Failed to generate embeddings for page ${page.id}: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    this.logger.log(
      `Embedding generation complete. Processed: ${processedCount}, Skipped: ${skippedCount}, Failed: ${pages.length - processedCount - skippedCount}`,
    );
  }

  /**
   * Generates embeddings for a single page.
   * Deletes existing embeddings for the page first, then creates new ones.
   */
  async generateEmbeddingsForPage(page: PageForEmbedding): Promise<void> {
    if (!page.textContent || page.textContent.trim().length === 0) {
      return;
    }

    // Delete existing embeddings for this page
    await this.deleteEmbeddingsForPage(page.id);

    // Split text into chunks
    const chunks = await this.textSplitter.splitText(page.textContent);

    if (chunks.length === 0) {
      return;
    }

    const model = this.getEmbeddingModel();
    const modelName = this.environmentService.getAiEmbeddingModel();
    const modelDimensions = this.getEmbeddingDimension();

    // Process chunks in batches to avoid API rate limits
    for (let i = 0; i < chunks.length; i += EMBEDDING_BATCH_SIZE) {
      const batchChunks = chunks.slice(i, i + EMBEDDING_BATCH_SIZE);

      const { embeddings } = await embedMany({
        model,
        values: batchChunks,
      });

      const rows = batchChunks.map((chunk, batchIndex) => {
        const chunkIndex = i + batchIndex;
        return {
          pageId: page.id,
          spaceId: page.spaceId,
          workspaceId: page.workspaceId,
          modelName,
          modelDimensions,
          chunkIndex,
          chunkStart: 0,
          chunkLength: chunk.length,
          metadata: JSON.stringify({ text: chunk }),
        };
      });

      // Insert rows one at a time with the vector value via raw SQL,
      // since Kysely doesn't natively support pgvector types.
      for (let j = 0; j < rows.length; j++) {
        const row = rows[j];
        const vectorStr = toSql(embeddings[j]);

        await sql`
          INSERT INTO page_embeddings (
            page_id, space_id, workspace_id, model_name, model_dimensions,
            embedding, chunk_index, chunk_start, chunk_length, metadata
          ) VALUES (
            ${row.pageId}, ${row.spaceId}, ${row.workspaceId},
            ${row.modelName}, ${row.modelDimensions},
            ${vectorStr}::vector, ${row.chunkIndex}, ${row.chunkStart},
            ${row.chunkLength}, ${row.metadata}::jsonb
          )
        `.execute(this.db);
      }
    }

    this.logger.debug(
      `Generated ${chunks.length} embeddings for page ${page.id}`,
    );
  }

  /**
   * Deletes all embeddings for a specific page.
   */
  async deleteEmbeddingsForPage(pageId: string): Promise<void> {
    await sql`
      DELETE FROM page_embeddings WHERE page_id = ${pageId}
    `.execute(this.db);
  }

  /**
   * Deletes all embeddings for a workspace.
   */
  async deleteEmbeddingsForWorkspace(workspaceId: string): Promise<void> {
    await sql`
      DELETE FROM page_embeddings WHERE workspace_id = ${workspaceId}
    `.execute(this.db);

    this.logger.log(`Deleted all embeddings for workspace: ${workspaceId}`);
  }
}
