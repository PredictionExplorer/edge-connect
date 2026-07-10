import { describe, expect, it } from 'vitest';
import { StarAiError, asStarAiError } from '../errors';

describe('AI error normalization', () => {
  it('preserves typed errors without wrapping them', () => {
    const error = new StarAiError('network', 'offline', true);
    expect(asStarAiError(error)).toBe(error);
  });

  it('maps AbortError to a non-retryable cancellation', () => {
    const normalized = asStarAiError(new DOMException('aborted', 'AbortError'));
    expect(normalized).toMatchObject({
      name: 'StarAiError',
      code: 'cancelled',
      retryable: false,
      message: 'AI request cancelled.',
    });
  });

  it('retains Error messages and unknown causes for diagnostics', () => {
    const cause = new TypeError('bad tensor');
    const normalized = asStarAiError(cause);
    expect(normalized).toMatchObject({
      code: 'internal',
      retryable: true,
      message: 'bad tensor',
      cause,
    });

    const unknown = { detail: 'opaque' };
    const fallback = asStarAiError(unknown);
    expect(fallback.message).toBe('AI request failed.');
    expect(fallback.cause).toBe(unknown);
  });
});

