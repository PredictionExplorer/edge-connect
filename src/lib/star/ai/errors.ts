export type StarAiErrorCode =
  | 'unavailable'
  | 'timeout'
  | 'network'
  | 'protocol'
  | 'stale'
  | 'illegal'
  | 'cancelled'
  | 'internal';

export class StarAiError extends Error {
  readonly code: StarAiErrorCode;
  readonly retryable: boolean;

  constructor(code: StarAiErrorCode, message: string, retryable = false, cause?: unknown) {
    super(message, { cause });
    this.name = 'StarAiError';
    this.code = code;
    this.retryable = retryable;
  }
}

export function asStarAiError(error: unknown): StarAiError {
  if (error instanceof StarAiError) return error;
  if (
    typeof DOMException !== 'undefined' &&
    error instanceof DOMException &&
    error.name === 'AbortError'
  ) {
    return new StarAiError('cancelled', 'AI request cancelled.');
  }
  return new StarAiError(
    'internal',
    error instanceof Error ? error.message : 'AI request failed.',
    true,
    error,
  );
}
