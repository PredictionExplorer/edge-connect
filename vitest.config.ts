import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    tsconfigPaths: true,
  },
  test: {
    clearMocks: true,
    restoreMocks: true,
    unstubGlobals: true,
    testTimeout: 15_000,
    projects: [
      {
        extends: true,
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.{test,spec}.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'components',
          environment: 'jsdom',
          include: ['src/**/*.{test,spec}.tsx'],
          setupFiles: ['./vitest.setup.ts'],
        },
      },
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary', 'html'],
      include: ['src/**/*.{ts,tsx}'],
      exclude: [
        'src/**/*.d.ts',
        'src/**/*.{test,spec}.{ts,tsx}',
        'src/**/__tests__/**',
        'src/app/layout.tsx',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 79,
        'src/lib/star/{board,game,scoring,symmetry}.ts': {
          lines: 90,
          functions: 85,
          branches: 85,
          statements: 90,
        },
        'src/lib/star/ai/**.ts': {
          lines: 80,
          functions: 80,
          branches: 75,
          statements: 78,
        },
        'src/components/**.tsx': {
          lines: 85,
          functions: 80,
          branches: 75,
          statements: 82,
        },
        'src/lib/store.ts': {
          lines: 85,
          functions: 80,
          branches: 80,
          statements: 84,
        },
        'src/workers/star-ai.worker.ts': {
          lines: 25,
          functions: 40,
          branches: 35,
          statements: 28,
        },
      },
    },
  },
});
