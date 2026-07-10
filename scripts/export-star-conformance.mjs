#!/usr/bin/env node

import { register } from 'node:module';

register(new URL('./typescript-loader.mjs', import.meta.url));
await import('./export-star-conformance.ts');
