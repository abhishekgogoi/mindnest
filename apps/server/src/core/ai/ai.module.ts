import { Module } from '@nestjs/common';
import { EmbeddingService } from './ai-embedding.service';
import { EmbeddingProcessor } from './ai-embedding.processor';
import { AiSearchService } from './ai-search.service';
import { AiSearchController } from './ai-search.controller';

@Module({
  controllers: [AiSearchController],
  providers: [EmbeddingService, EmbeddingProcessor, AiSearchService],
  exports: [EmbeddingService, AiSearchService],
})
export class AiModule {}
