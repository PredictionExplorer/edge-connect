/** @type {import('@stryker-mutator/api/core').PartialStrykerOptions} */
const config = {
  testRunner: 'vitest',
  coverageAnalysis: 'perTest',
  ignoreStatic: true,
  concurrency: 4,
  ignorePatterns: [
    '.git/**',
    '.next/**',
    '.stryker-tmp/**',
    'coverage/**',
    'node_modules/**',
    'playwright-report/**',
    'public/**',
    'test-results/**',
    'training/**',
  ],
  mutate: [
    'src/lib/star/game.ts',
    'src/lib/star/scoring.ts',
    'src/lib/star/symmetry.ts',
    'src/lib/star/ai/protocol.ts',
  ],
  reporters: ['progress', 'clear-text', 'html'],
  thresholds: {
    high: 85,
    low: 75,
    break: 70,
  },
  vitest: {
    configFile: 'vitest.config.ts',
    related: true,
  },
};

export default config;

