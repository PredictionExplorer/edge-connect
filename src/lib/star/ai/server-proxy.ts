export const STAR_AI_PROXY_MOVE_PATH = '/v1/move' as const;
export const STAR_AI_PROXY_HEALTH_PATH = '/v1/health' as const;
export const STAR_AI_PROXY_REQUEST_BYTES = 64 * 1024;
export const STAR_AI_PROXY_RESPONSE_BYTES = 1024 * 1024;
export const STAR_AI_PROXY_MOVE_TIMEOUT_MS = 60_000;
export const STAR_AI_PROXY_HEALTH_TIMEOUT_MS = 5_000;

const REQUEST_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export interface StarAiProxyConfig {
  serverUrl: string | undefined;
  bearerToken?: string;
  moveTimeoutMs?: number;
  healthTimeoutMs?: number;
}

class BodyLimitError extends Error {}

function noStoreHeaders(requestId: string): HeadersInit {
  return {
    'Cache-Control': 'no-store, max-age=0',
    'Content-Type': 'application/json; charset=utf-8',
    Pragma: 'no-cache',
    'X-Content-Type-Options': 'nosniff',
    'X-Request-ID': requestId,
  };
}

function proxyError(
  requestId: string,
  status: number,
  code: string,
  message: string,
  retryable: boolean,
): Response {
  return Response.json(
    {
      error: {
        code,
        message,
        retryable,
        request_id: requestId,
      },
    },
    { status, headers: noStoreHeaders(requestId) },
  );
}

function requestIdFor(request: Request): string {
  const supplied = request.headers.get('X-Request-ID');
  if (supplied && REQUEST_ID.test(supplied)) return supplied;
  return crypto.randomUUID().replaceAll('-', '');
}

function isJsonContentType(value: string | null): boolean {
  return value?.split(';', 1)[0].trim().toLowerCase() === 'application/json';
}

function endpointBase(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, '');
  for (const suffix of [
    '/v1/move',
    '/v1/analyze',
    '/v1/health',
    '/healthz',
    '/v1',
  ]) {
    if (normalized.endsWith(suffix)) return normalized.slice(0, -suffix.length);
  }
  return normalized;
}

export function resolveStarAiUpstreamUrl(serverUrl: string, endpoint: string): string {
  let target: URL;
  try {
    target = new URL(serverUrl);
  } catch (error) {
    throw new Error('STAR_AI_SERVER_URL must be an absolute URL', { cause: error });
  }
  if (
    (target.protocol !== 'http:' && target.protocol !== 'https:') ||
    target.username ||
    target.password ||
    target.search ||
    target.hash
  ) {
    throw new Error('STAR_AI_SERVER_URL must be an HTTP(S) URL without credentials');
  }
  target.pathname = `${endpointBase(target.pathname)}${endpoint}`;
  return target.toString();
}

async function readLimitedBody(
  body: ReadableStream<Uint8Array> | null,
  contentLength: string | null,
  maximum: number,
): Promise<Uint8Array> {
  if (contentLength) {
    const declared = Number(contentLength);
    if (!Number.isSafeInteger(declared) || declared < 0 || declared > maximum) {
      throw new BodyLimitError('body exceeds limit');
    }
  }
  if (!body) return new Uint8Array();

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximum) {
        await reader.cancel();
        throw new BodyLimitError('body exceeds limit');
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

export async function proxyStarAiRequest(
  request: Request,
  endpoint: typeof STAR_AI_PROXY_MOVE_PATH | typeof STAR_AI_PROXY_HEALTH_PATH,
  config: StarAiProxyConfig = {
    serverUrl: process.env.STAR_AI_SERVER_URL,
    bearerToken: process.env.STAR_AI_BEARER_TOKEN,
  },
): Promise<Response> {
  const requestId = requestIdFor(request);
  if (!config.serverUrl?.trim()) {
    return proxyError(
      requestId,
      503,
      'star_ai_unavailable',
      'Server AI is not configured.',
      false,
    );
  }

  let target: string;
  try {
    target = resolveStarAiUpstreamUrl(config.serverUrl, endpoint);
  } catch {
    return proxyError(
      requestId,
      503,
      'star_ai_unavailable',
      'Server AI configuration is invalid.',
      false,
    );
  }

  let body: Uint8Array | undefined;
  if (endpoint === STAR_AI_PROXY_MOVE_PATH) {
    if (!isJsonContentType(request.headers.get('Content-Type'))) {
      return proxyError(
        requestId,
        415,
        'invalid_content_type',
        'Expected application/json.',
        false,
      );
    }
    try {
      body = await readLimitedBody(
        request.body,
        request.headers.get('Content-Length'),
        STAR_AI_PROXY_REQUEST_BYTES,
      );
      JSON.parse(new TextDecoder().decode(body));
    } catch (error) {
      return proxyError(
        requestId,
        error instanceof BodyLimitError ? 413 : 400,
        error instanceof BodyLimitError ? 'request_too_large' : 'invalid_json',
        error instanceof BodyLimitError ? 'Request body is too large.' : 'Request body is invalid JSON.',
        false,
      );
    }
  }

  const timeoutMs =
    endpoint === STAR_AI_PROXY_MOVE_PATH
      ? (config.moveTimeoutMs ?? STAR_AI_PROXY_MOVE_TIMEOUT_MS)
      : (config.healthTimeoutMs ?? STAR_AI_PROXY_HEALTH_TIMEOUT_MS);
  const controller = new AbortController();
  let timedOut = false;
  const abortFromClient = () => controller.abort(request.signal.reason);
  request.signal.addEventListener('abort', abortFromClient, { once: true });
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const headers = new Headers({
      Accept: 'application/json',
      'X-Request-ID': requestId,
    });
    if (body) headers.set('Content-Type', 'application/json');
    if (config.bearerToken) headers.set('Authorization', `Bearer ${config.bearerToken}`);
    const upstreamBody = body ? Uint8Array.from(body).buffer : undefined;

    const upstream = await fetch(target, {
      method: endpoint === STAR_AI_PROXY_MOVE_PATH ? 'POST' : 'GET',
      body: upstreamBody,
      headers,
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!isJsonContentType(upstream.headers.get('Content-Type'))) {
      return proxyError(
        requestId,
        502,
        'invalid_upstream_response',
        'Server AI returned an invalid response.',
        true,
      );
    }

    let payload: unknown;
    try {
      const responseBody = await readLimitedBody(
        upstream.body,
        upstream.headers.get('Content-Length'),
        STAR_AI_PROXY_RESPONSE_BYTES,
      );
      payload = JSON.parse(new TextDecoder().decode(responseBody));
    } catch {
      return proxyError(
        requestId,
        502,
        'invalid_upstream_response',
        'Server AI returned an invalid response.',
        true,
      );
    }

    if (upstream.status === 401 || upstream.status === 403 || upstream.status >= 500) {
      return proxyError(
        requestId,
        503,
        'star_ai_unavailable',
        'Server AI is unavailable.',
        true,
      );
    }
    const upstreamRequestId = upstream.headers.get('X-Request-ID');
    const responseRequestId =
      upstreamRequestId && REQUEST_ID.test(upstreamRequestId) ? upstreamRequestId : requestId;
    return Response.json(payload, {
      status: upstream.status,
      headers: noStoreHeaders(responseRequestId),
    });
  } catch {
    return proxyError(
      requestId,
      503,
      timedOut ? 'star_ai_timeout' : 'star_ai_unavailable',
      timedOut ? 'Server AI timed out.' : 'Server AI is unavailable.',
      true,
    );
  } finally {
    clearTimeout(timeout);
    request.signal.removeEventListener('abort', abortFromClient);
  }
}
