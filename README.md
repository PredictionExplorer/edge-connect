# ✳Star

A browser implementation of [*Star](https://en.wikipedia.org/wiki/*Star), Ea Ea's connection
game of peries, quarks and stars. Human play supports both variants; the AI stack targets
Double *Star only.

## Features

- **Both variants**: classic *Star (one stone per turn) and Double *Star (two stones per
  turn; the first player places a single stone on the opening turn).
- **Any board size**: the official Mini (4 rings), Small (6), Medium (8) and Full (10)
  boards, or any custom size from 3 to 12 rings.
- **Exact scoring, live**: peries (occupied and enclosed), the quark peri for holding three
  or more corners, the ±2 × star-difference award, and the quark tie-break — recomputed
  after every stone with a union-find + flood-fill engine over a CSR adjacency (microseconds
  per evaluation, so the score panel is always current).
- **Table manners**: pass to agree the score, undo/redo, optional pie rule, influence
  overlay showing who claims each peri, and a full score reveal when the sky is settled.
- Games persist in `localStorage`, so a refresh resumes play.

## Double *Star AI

The repository includes a self-play training and inference stack for no-pie Double *Star
on boards with 3–12 rings:

- Rust rules, scoring, symmetry and Gumbel tree search, exposed to Python as
  `star_native`;
- a PyTorch graph ResTNet with policy, WDL, score, ownership and alive heads;
- replay, learner, candidate/champion arenas and single-host 4/8-H100 orchestration;
- a private GPU `starserve` backend and a distilled ONNX + Rust/WASM browser runtime.

No trained model is checked in. Server and local AI choices remain unavailable until an
operator trains a champion or publishes a distilled browser model. This is implemented
training infrastructure, not a claim of superhuman playing strength. See the
[training operator guide](training/README.md) and
[serving/distillation details](training/docs/serving-and-distillation.md).

## Development

```bash
npm ci
npm run dev              # development server at http://localhost:3000
npm test                 # Vitest suite
npm run lint             # ESLint
npm run build            # production Next.js build
npm run start            # serve the production build
npm run build:star-wasm  # optional local-AI Rust/WASM artifacts
```

The rules engine lives in `src/lib/star/`:

- `board.ts` — pentagonal mesh generation, official Nxy notation, CSR adjacency, layout
- `scoring.ts` — the scoring engine (see the file header for the exact rule semantics)
- `game.ts` — turn protocol for both variants, passes, pie rule, replayable action log

Scoring is cross-validated in tests against an independent naive reference implementation
on thousands of random positions, plus the classic invariant: on a decided board the two
totals always sum to the number of peries + 1.

## Deploying to Vercel

```bash
npm i -g vercel
vercel         # preview
vercel --prod  # production
```

Human play and published local-AI assets need no private service. Server AI uses the
same-origin Next.js routes at `/v1/move` and `/v1/health`; configure the deployment with
server-only `STAR_AI_SERVER_URL` and, when enabled by `starserve`,
`STAR_AI_BEARER_TOKEN`. Never expose the bearer token through a `NEXT_PUBLIC_*`
variable.
