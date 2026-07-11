import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const crate = resolve(root, 'training/crates/star-wasm');
const wasmDirectory = 'wasm-2da3783519381453';
const output = resolve(root, `public/models/star/${wasmDirectory}`);
const outputFromCrate = relative(crate, output);

if (!existsSync(resolve(crate, 'Cargo.toml'))) {
  throw new Error(`star-wasm crate not found at ${crate}`);
}

mkdirSync(dirname(output), { recursive: true });
rmSync(output, { recursive: true, force: true });

const result = spawnSync(
  'wasm-pack',
  [
    'build',
    '.',
    '--target',
    'web',
    '--release',
    '--out-dir',
    outputFromCrate,
    '--out-name',
    'star_wasm',
    '--no-pack',
    '--',
    '--locked',
  ],
  {
    cwd: crate,
    env: {
      ...process.env,
      SOURCE_DATE_EPOCH: process.env.SOURCE_DATE_EPOCH ?? '0',
    },
    stdio: 'inherit',
  },
);

if (result.error) throw result.error;
if (result.status !== 0) {
  throw new Error(`wasm-pack exited with status ${String(result.status)}`);
}

for (const filename of ['star_wasm.js', 'star_wasm_bg.wasm']) {
  if (!existsSync(resolve(output, filename))) {
    throw new Error(`wasm-pack did not produce ${filename}`);
  }
}

writeFileSync(
  resolve(output, 'contract.json'),
  `${JSON.stringify(
    {
      schema: 'edgeconnect.star.browser-wasm-build.v2',
      rulesSchema: 'edgeconnect.star.rules.v2',
      rulesHash: 'fnv1a64:2da3783519381453',
      moduleUrl: `/models/star/${wasmDirectory}/star_wasm.js`,
      binaryUrl: `/models/star/${wasmDirectory}/star_wasm_bg.wasm`,
      modelManifestUrl: '/models/star/manifest.json',
    },
    null,
    2,
  )}\n`,
);

console.log(`Built Star WASM assets in public/models/star/${wasmDirectory}`);
console.log('Browser model manifest convention: public/models/star/manifest.json');
