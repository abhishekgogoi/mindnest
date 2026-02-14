import { IsNotEmpty, IsOptional, IsString } from 'class-validator';

export class AiSearchDTO {
  @IsNotEmpty()
  @IsString()
  query: string;

  @IsOptional()
  @IsString()
  spaceId?: string;
}
