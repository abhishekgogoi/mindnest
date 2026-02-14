import { Logger, OnModuleDestroy } from '@nestjs/common';
import { OnWorkerEvent, Processor, WorkerHost } from '@nestjs/bullmq';
import { Job } from 'bullmq';
import { QueueJob, QueueName } from '../../integrations/queue/constants';
import { EmbeddingService } from './ai-embedding.service';
import { InjectKysely } from 'nestjs-kysely';
import { KyselyDB } from '@docmost/db/types/kysely.types';

interface IWorkspaceEmbeddingsJob {
  workspaceId: string;
}

interface IPageEmbeddingsJob {
  pageId: string;
  workspaceId: string;
}

@Processor(QueueName.AI_QUEUE)
export class EmbeddingProcessor extends WorkerHost implements OnModuleDestroy {
  private readonly logger = new Logger(EmbeddingProcessor.name);

  constructor(
    private readonly embeddingService: EmbeddingService,
    @InjectKysely() private readonly db: KyselyDB,
  ) {
    super();
  }

  async process(
    job: Job<IWorkspaceEmbeddingsJob | IPageEmbeddingsJob>,
  ): Promise<void> {
    switch (job.name) {
      case QueueJob.WORKSPACE_CREATE_EMBEDDINGS:
        await this.handleWorkspaceCreateEmbeddings(job.data.workspaceId);
        break;

      case QueueJob.WORKSPACE_DELETE_EMBEDDINGS:
        await this.handleWorkspaceDeleteEmbeddings(job.data.workspaceId);
        break;

      case QueueJob.GENERATE_PAGE_EMBEDDINGS:
        if ('pageId' in job.data) {
          await this.handleGeneratePageEmbeddings(job.data.pageId);
        }
        break;

      case QueueJob.DELETE_PAGE_EMBEDDINGS:
        if ('pageId' in job.data) {
          await this.handleDeletePageEmbeddings(job.data.pageId);
        }
        break;

      default:
        this.logger.warn(`Unknown job name: ${job.name}`);
    }
  }

  private async handleWorkspaceCreateEmbeddings(
    workspaceId: string,
  ): Promise<void> {
    await this.embeddingService.generateEmbeddingsForWorkspace(workspaceId);
  }

  private async handleWorkspaceDeleteEmbeddings(
    workspaceId: string,
  ): Promise<void> {
    await this.embeddingService.deleteEmbeddingsForWorkspace(workspaceId);
  }

  private async handleGeneratePageEmbeddings(pageId: string): Promise<void> {
    const page = await this.db
      .selectFrom('pages')
      .select(['id', 'spaceId', 'workspaceId', 'textContent'])
      .where('id', '=', pageId)
      .where('deletedAt', 'is', null)
      .executeTakeFirst();

    if (!page) {
      this.logger.warn(`Page not found for embedding: ${pageId}`);
      return;
    }

    await this.embeddingService.generateEmbeddingsForPage(page);
  }

  private async handleDeletePageEmbeddings(pageId: string): Promise<void> {
    await this.embeddingService.deleteEmbeddingsForPage(pageId);
  }

  @OnWorkerEvent('active')
  onActive(job: Job): void {
    this.logger.debug(`Processing ${job.name} job`);
  }

  @OnWorkerEvent('failed')
  onFailed(job: Job): void {
    this.logger.error(
      `Error processing ${job.name} job. Reason: ${job.failedReason}`,
    );
  }

  @OnWorkerEvent('completed')
  onCompleted(job: Job): void {
    this.logger.debug(`Completed ${job.name} job`);
  }

  async onModuleDestroy(): Promise<void> {
    if (this.worker) {
      await this.worker.close();
    }
  }
}
