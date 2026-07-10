import {
  STAR_AI_PROXY_HEALTH_PATH,
  proxyStarAiRequest,
} from '@/lib/star/ai/server-proxy';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 10;

export async function GET(request: Request): Promise<Response> {
  return proxyStarAiRequest(request, STAR_AI_PROXY_HEALTH_PATH);
}
