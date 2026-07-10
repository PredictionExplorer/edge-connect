# ✳Star

A browser implementation of [*Star](https://en.wikipedia.org/wiki/*Star), Ea Ea's connection
game of peries, quarks and stars — for two players sharing one screen.

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

## Development

```bash
npm install
npm run dev        # http://localhost:3000
npm test           # engine test suite (board, scoring, game protocol)
npm run build      # production build
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
vercel           # preview
vercel --prod    # production
```

No environment variables or server components required — the whole game is a static,
client-side Next.js app.
