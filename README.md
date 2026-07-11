# ✳Star

A browser implementation of [*Star](https://en.wikipedia.org/wiki/*Star), Ea Ea's connection
game of peries, quarks and stars. Human play supports both variants; the AI stack targets
Double *Star only.

## Features

- **Both variants**: classic *Star (one stone per turn) and Double *Star (two stones per
  turn; the first player places a single stone on the opening turn).
- **Four supported boards**: official Mini (4 rings), Small (6), Medium (8), and
  Full (10). Other ring counts are rejected at every rules and API boundary.
- **Exact scoring, live**: peries (occupied and enclosed), the quark peri for holding three
  or more corners, the ±2 × star-difference award, and the quark tie-break — recomputed
  after every stone with a union-find + flood-fill engine over a CSR adjacency (microseconds
  per evaluation, so the score panel is always current).
- **Complete games**: placement is mandatory until the board is full, with undo/redo,
  optional web-only pie rule, influence overlays, and a final score reveal.
- Games persist in `localStorage`, so a refresh resumes play.

## Double *Star AI

The repository includes a self-play training and inference stack for no-pie Double *Star
on 4-, 6-, 8-, and 10-ring boards:

- Rust rules, scoring, symmetry and Gumbel tree search, exposed to Python as
  `star_native`;
- a PyTorch graph ResTNet with node policy, binary outcome, score, ownership, and
  alive heads;
- replay, learner, candidate/champion arenas and single-host 4/8-H100 orchestration;
- a private GPU `starserve` backend and a distilled ONNX + Rust/WASM browser runtime.

No trained model is checked in. Server and local AI choices remain unavailable until an
operator trains a champion or publishes a distilled browser model. This is implemented
training infrastructure, not a claim of superhuman playing strength. See the
[training operator guide](training/README.md) and
[production H100 training runbook](training/docs/production-h100-training-runbook.md), then the
[target-host benchmark results](training/docs/h100-target-host-benchmark-results.md) and
[serving/distillation details](training/docs/serving-and-distillation.md).

## Development

```bash
npm ci
npm run dev              # development server at http://localhost:3000
npm test                 # Vitest suite
npm run test:coverage    # unit/component coverage with risk-weighted floors
npm run test:e2e         # production build + Chromium/Firefox/WebKit flows
npm run lint             # ESLint
npm run typecheck        # strict TypeScript
npm run build            # production Next.js build
npm run start            # serve the production build
npm run build:star-wasm  # optional local-AI Rust/WASM artifacts
node scripts/export-star-conformance.mjs # regenerate conformance-v2.json
```

CI also builds the native Python extension, runs pytest with per-module coverage
floors, Rust property and WASM contract suites, mutation jobs, dependency audits,
and a container smoke. CUDA, NCCL, and soak tests are separate hardware tiers; see
[testing and H100 validation](training/docs/testing-and-h100-validation.md).

The rules engine lives in `src/lib/star/`:

- `board.ts` — pentagonal mesh generation, official Nxy notation, CSR adjacency, layout
- `scoring.ts` — the scoring engine (see the file header for the exact rule semantics)
- `game.ts` — placement-only turn protocol, web pie rule, replayable action log

Scoring is cross-validated against an independent reference implementation. Full supported
boards have no contested peries, totals sum to `5 × rings + 1`, and the odd nonzero margin
always identifies exactly one winner.

## Deploying to Vercel

```bash
npm i -g vercel
vercel         # preview
vercel --prod  # production
```

Human play and published local-AI assets need no private service. Server AI uses
same-origin Next.js routes at `/v2/move`, `/v2/analyze`, and `/v2/health`; configure
the deployment with server-only `STAR_AI_SERVER_URL` and, when enabled by `starserve`,
`STAR_AI_BEARER_TOKEN`. Never expose the bearer token through a `NEXT_PUBLIC_*`
variable.
