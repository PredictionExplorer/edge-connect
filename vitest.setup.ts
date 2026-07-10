import '@testing-library/jest-dom/vitest';
import 'vitest-axe/extend-expect';
import { cleanup } from '@testing-library/react';
import * as axeMatchers from 'vitest-axe/matchers';
import { afterEach, expect } from 'vitest';

expect.extend(axeMatchers);
afterEach(cleanup);

Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
  configurable: true,
  value: () => null,
});

