import { Injectable, Logger } from '@nestjs/common';
import { InjectKysely } from 'nestjs-kysely';
import { KyselyDB } from '@docmost/db/types/kysely.types';
import { EnvironmentService } from '../../integrations/environment/environment.service';
import { embed, streamText, LanguageModel } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { toSql } from 'pgvector';
import { sql } from 'kysely';
import { EmbeddingService } from './ai-embedding.service';
import { SpaceMemberRepo } from '@docmost/db/repos/space/space-member.repo';

/** Maximum number of similar chunks to retrieve for context. */
const TOP_K = 10;

interface SearchChunkResult {
  pageId: string;
  spaceId: string;
  chunkIndex: number;
  chunkText: string;
  distance: number;
}

interface SourceResult {
  pageId: string;
  title: string;
  slugId: string;
  spaceSlug: string;
  similarity: number;
  distance: number;
  chunkIndex: number;
  excerpt: string;
}

@Injectable()
export class AiSearchService {
  private readonly logger = new Logger(AiSearchService.name);

  constructor(
    @InjectKysely() private readonly db: KyselyDB,
    private readonly environmentService: EnvironmentService,
    private readonly embeddingService: EmbeddingService,
    private readonly spaceMemberRepo: SpaceMemberRepo,
  ) {}

  /**
   * Creates the language model instance based on the configured AI driver.
   */
  private getLanguageModel(): LanguageModel {
    const driver = this.environmentService.getAiDriver();
    const modelName = this.environmentService.getAiCompletionModel();

    switch (driver) {
      case 'openai': {
        const openai = createOpenAI({
          apiKey: this.environmentService.getOpenAiApiKey(),
        });
        return openai(modelName);
      }
      case 'gemini': {
        const google = createGoogleGenerativeAI({
          apiKey: this.environmentService.getGeminiApiKey(),
        });
        return google.languageModel(modelName);
      }
      case 'ollama': {
        const ollama = createOpenAICompatible({
          name: 'ollama',
          baseURL: this.environmentService.getOllamaApiUrl() + '/v1',
        });
        return ollama.languageModel(modelName);
      }
      default:
        throw new Error(`Unsupported AI driver: ${driver}`);
    }
  }

  /**
   * Finds the most similar embedding chunks to the query vector,
   * scoped to the user's accessible spaces within the workspace.
   */
  private async findSimilarChunks(
    queryVector: number[],
    workspaceId: string,
    userId: string,
    spaceId?: string,
  ): Promise<SearchChunkResult[]> {
    const vectorStr = toSql(queryVector);

    // Resolve the set of space IDs the user may search.
    let accessibleSpaceIds: string[];

    if (spaceId) {
      accessibleSpaceIds = [spaceId];
    } else {
      const rows = await this.spaceMemberRepo
        .getUserSpaceIdsQuery(userId)
        .execute();
      accessibleSpaceIds = rows.map((r) => r.id);
    }

    if (accessibleSpaceIds.length === 0) {
      return [];
    }

    // Format as a PostgreSQL array literal so it can be passed
    // as a parameterized value (not raw SQL). Kysely's tagged
    // template will bind this as $N, preventing SQL injection.
    const pgArrayLiteral = `{${accessibleSpaceIds.join(',')}}`;

    const results = await sql<SearchChunkResult>`
      SELECT
        pe.page_id AS "pageId",
        pe.space_id AS "spaceId",
        pe.chunk_index AS "chunkIndex",
        (pe.metadata #>> '{}')::jsonb->>'text' AS "chunkText",
        (pe.embedding <=> ${vectorStr}::vector) AS "distance"
      FROM page_embeddings pe
      WHERE pe.workspace_id = ${workspaceId}
        AND pe.space_id = ANY(${pgArrayLiteral}::uuid[])
      ORDER BY pe.embedding <=> ${vectorStr}::vector
      LIMIT ${TOP_K}
    `.execute(this.db);

    return results.rows;
  }

  /**
   * Enriches raw chunk results with page metadata (title, slugId, space slug)
   * and returns them as source references for the client.
   */
  private async buildSources(
    chunks: SearchChunkResult[],
  ): Promise<SourceResult[]> {
    if (chunks.length === 0) {
      return [];
    }

    const pageIds = [...new Set(chunks.map((c) => c.pageId))];

    const pages = await this.db
      .selectFrom('pages')
      .innerJoin('spaces', 'spaces.id', 'pages.spaceId')
      .select([
        'pages.id',
        'pages.title',
        'pages.slugId',
        'spaces.slug as spaceSlug',
      ])
      .where('pages.id', 'in', pageIds)
      .execute();

    const pageMap = new Map(pages.map((p) => [p.id, p]));

    return chunks
      .map((chunk) => {
        const page = pageMap.get(chunk.pageId);
        if (!page) return null;

        return {
          pageId: chunk.pageId,
          title: page.title || 'Untitled',
          slugId: page.slugId,
          spaceSlug: page.spaceSlug,
          similarity: 1 - chunk.distance,
          distance: chunk.distance,
          chunkIndex: chunk.chunkIndex,
          excerpt: (chunk.chunkText ?? '').slice(0, 200),
        };
      })
      .filter((s): s is SourceResult => s !== null);
  }

  /**
   * Builds the system prompt with retrieved context chunks
   * to ground the LLM answer in the workspace content.
   */
  private buildSystemPrompt(chunks: SearchChunkResult[]): string {
    const contextBlock = chunks
      .map((c, i) => `[${i + 1}] ${c.chunkText}`)
      .join('\n\n');

    return [
      'You are a helpful AI assistant that answers questions based on the user\'s workspace documents.',
      'Use ONLY the following context to answer. If the context does not contain enough information, say so clearly.',
      'Keep your answer concise and well-structured. Use markdown formatting where appropriate.',
      '',
      '--- Context ---',
      contextBlock,
      '--- End Context ---',
    ].join('\n');
  }

  /**
   * Main entry point: embeds the question, retrieves relevant chunks,
   * and streams the LLM answer. Yields SSE-formatted chunks.
   */
  async *askQuestion(
    query: string,
    workspaceId: string,
    userId: string,
    spaceId?: string,
  ): AsyncGenerator<string> {
    // Step 1: Embed the user's question using the same embedding model
    // that was used to generate page embeddings.
    const embeddingModel = this.embeddingService.getEmbeddingModel();

    const { embedding: queryVector } = await embed({
      model: embeddingModel,
      value: query,
    });

    // Step 2: Find the most similar chunks in the workspace
    const chunks = await this.findSimilarChunks(
      queryVector,
      workspaceId,
      userId,
      spaceId,
    );

    if (chunks.length === 0) {
      yield `data: ${JSON.stringify({ content: 'No relevant content found in your workspace for this query.' })}\n\n`;
      yield `data: ${JSON.stringify({ sources: [] })}\n\n`;
      yield 'data: [DONE]\n\n';
      return;
    }

    // Step 3: Stream the LLM answer with retrieved context
    const systemPrompt = this.buildSystemPrompt(chunks);
    const languageModel = this.getLanguageModel();

    const result = streamText({
      model: languageModel,
      system: systemPrompt,
      prompt: query,
    });

    for await (const textPart of result.textStream) {
      yield `data: ${JSON.stringify({ content: textPart })}\n\n`;
    }

    // Step 4: Send sources after the answer stream completes
    const sources = await this.buildSources(chunks);
    yield `data: ${JSON.stringify({ sources })}\n\n`;
    yield 'data: [DONE]\n\n';
  }
}
