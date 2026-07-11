/**
 * Export deterministic *Star vectors for cross-language parity.
 *
 * Usage:
 *   node scripts/export-star-conformance.mjs [output-path]
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serializeStarConformance } from '../src/lib/star/conformance';
import {
  fnv1a64,
  STAR_RULES_CANONICAL,
  STAR_RULES_HASH,
  STAR_RULES_HASH_ALGORITHM,
} from '../src/lib/star/rules';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, '..');
const outputPath = process.argv[2]
  ? resolve(process.cwd(), process.argv[2])
  : resolve(repositoryRoot, 'testdata/star/conformance-v2.json');

const actualHash =
  `${STAR_RULES_HASH_ALGORITHM}:${fnv1a64(STAR_RULES_CANONICAL)}`;
if (actualHash !== STAR_RULES_HASH) {
  throw new Error(
    `stale STAR_RULES_HASH: expected ${actualHash}, found ${STAR_RULES_HASH}`,
  );
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, serializeStarConformance(), 'utf8');
console.log(`Wrote ${outputPath}`);
