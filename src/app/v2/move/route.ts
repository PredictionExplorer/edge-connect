import {
  STAR_AI_PROXY_MOVE_PATH,
  proxyStarAiRequest,
} from '@/lib/star/ai/server-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 65;

export async function POST(request: Request): Promise<Response> {
  return proxyStarAiRequest(request, STAR_AI_PROXY_MOVE_PATH);
}
