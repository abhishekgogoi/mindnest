import {
  Body,
  Controller,
  HttpCode,
  HttpStatus,
  Logger,
  Post,
  Res,
  UseGuards,
} from '@nestjs/common';
import { AiSearchService } from './ai-search.service';
import { AiSearchDTO } from './dto/ai-search.dto';
import { AuthUser } from '../../common/decorators/auth-user.decorator';
import { AuthWorkspace } from '../../common/decorators/auth-workspace.decorator';
import { JwtAuthGuard } from '../../common/guards/jwt-auth.guard';
import { User, Workspace } from '@docmost/db/types/entity.types';
import { FastifyReply } from 'fastify';

@UseGuards(JwtAuthGuard)
@Controller('ai')
export class AiSearchController {
  private readonly logger = new Logger(AiSearchController.name);

  constructor(private readonly aiSearchService: AiSearchService) {}

  @HttpCode(HttpStatus.OK)
  @Post('ask')
  async ask(
    @Body() dto: AiSearchDTO,
    @AuthUser() user: User,
    @AuthWorkspace() workspace: Workspace,
    @Res() res: FastifyReply,
  ): Promise<void> {
    res.raw.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });

    try {
      const stream = this.aiSearchService.askQuestion(
        dto.query,
        workspace.id,
        user.id,
        dto.spaceId,
      );

      for await (const chunk of stream) {
        res.raw.write(chunk);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'An unexpected error occurred';
      this.logger.error(`AI search failed: ${message}`);
      res.raw.write(
        `data: ${JSON.stringify({ error: message })}\n\n`,
      );
    }

    res.raw.end();
  }
}
