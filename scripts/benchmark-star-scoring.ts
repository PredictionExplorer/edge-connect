import { performance } from 'node:perf_hooks';
import { getBoard } from '../src/lib/star/board';
import { scorePosition } from '../src/lib/star/scoring';

const rings = Number(process.env.STAR_BENCH_RINGS ?? 10);
const iterations = Number(process.env.STAR_BENCH_ITERATIONS ?? 50_000);
const maximumMicros = Number(process.env.STAR_BENCH_MAX_US ?? Number.POSITIVE_INFINITY);
if (
  !Number.isInteger(rings) ||
  rings < 3 ||
  rings > 12 ||
  !Number.isInteger(iterations) ||
  iterations <= 0 ||
  maximumMicros <= 0
) {
  throw new Error('invalid STAR_BENCH_RINGS, STAR_BENCH_ITERATIONS, or STAR_BENCH_MAX_US');
}

let randomState = 0xbe5eeed;
const random = () => {
  randomState = (randomState + 0x6d2b79f5) | 0;
  let value = Math.imul(randomState ^ (randomState >>> 15), 1 | randomState);
  value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
};

const board = getBoard(rings);
const positions = Array.from({ length: 128 }, () =>
  Int8Array.from({ length: board.n }, () => (random() < 0.5 ? 0 : 1)),
);
for (const stones of positions) scorePosition(board, stones);

let checksum = 0;
const started = performance.now();
for (let index = 0; index < iterations; index++) {
  checksum += scorePosition(board, positions[index % positions.length]).players[0].total;
}
const elapsedMs = performance.now() - started;
const microsecondsPerPosition = (elapsedMs * 1_000) / iterations;
const result = {
  schemaVersion: 1,
  benchmark: 'typescript-star-scoring',
  rings,
  nodes: board.n,
  iterations,
  elapsedMs,
  microsecondsPerPosition,
  positionsPerSecond: iterations / (elapsedMs / 1_000),
  checksum,
  maximumMicrosecondsPerPosition: Number.isFinite(maximumMicros) ? maximumMicros : null,
  passed: microsecondsPerPosition <= maximumMicros,
};
console.log(JSON.stringify(result));
if (!result.passed) process.exitCode = 2;

