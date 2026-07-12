'use client';

import { useId, useRef } from 'react';
import { X } from 'lucide-react';
import { ModalDialog } from './ModalDialog';

interface RulesDialogProps {
  open: boolean;
  onClose: () => void;
}

export function RulesDialog({ open, onClose }: RulesDialogProps) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const titleId = useId();

  return (
    <ModalDialog
      open={open}
      onClose={onClose}
      ariaLabel="How to play *Star"
      initialFocusRef={closeButton}
      className="max-w-2xl"
    >
      <article
        className="thin-scroll panel-surface relative max-h-[calc(100dvh-2rem)] w-full overflow-y-auto rounded-3xl p-5 shadow-2xl sm:p-8"
      >
        <button
          ref={closeButton}
          type="button"
          onClick={onClose}
          aria-label="Close rules"
          className="absolute right-4 top-4 rounded-full border border-white/10 p-2 text-muted transition-colors hover:border-gold/40 hover:text-ink"
        >
          <X className="h-4 w-4" />
        </button>

        <h2 id={titleId} className="font-display pr-12 text-3xl text-gold-strong">
          How to play
        </h2>

        <div className="mt-4 space-y-4 text-sm leading-relaxed text-ink/90">
          <p>
            Two players take turns placing stones on empty nodes — one stone per turn in{' '}
            <em>*Star</em>, two per turn in <em>Double *Star</em> (the first player places just
            one stone on the game&apos;s very first turn). A placement is mandatory while an
            empty node remains, and stones never move. Boards use 4, 6, 8, or 10 rings. The
            star-shaped <strong className="text-gold">bridge</strong> in the center cannot be
            played, but it connects all five innermost nodes for <em>both</em> players.
          </p>
          <p>
            Every node on the perimeter holds a <strong className="text-gold">peri</strong>,
            worth one point. The five corners each hold a{' '}
            <strong className="text-gold">quark</strong> as well.
          </p>
          <p>
            A connected group of your stones that occupies at least two peries is a{' '}
            <strong className="text-gold">star</strong>. A star owns the peries it occupies, plus
            any peries it walls off from the rest of the board. Groups that fail to make a star
            are removed at scoring time — the peries around them go to whoever surrounds them.
            Stones are never physically removed. A whole group is crossed out early only when it
            cannot reach two peries even if every remaining open node were assigned to its color.
          </p>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted">
              Final score
            </h3>
            <ul className="space-y-1.5">
              <li>+1 for each peri you own</li>
              <li>
                +1 <em>quark peri</em> if you own three or more of the five quarks
              </li>
              <li>
                ±2 × the difference in star counts — the player with <em>fewer</em> stars is
                rewarded, the player with more is penalized
              </li>
            </ul>
          </div>
          <p>
            So two stones grabbing two peries look like two points, but as a separate star they
            cancel out — unless they split the opponent or claim a decisive quark. Connect
            everything; waste nothing. Live totals may be tied, but on a full board the two
            totals sum to the number of peries plus one. That sum is odd on every supported
            board, so the final margin is nonzero and someone always wins.
          </p>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted">
              Completion bounds
            </h3>
            <p>
              The score panel also fills every open node with one color, then the other, to show
              the two extreme final scores. These are mathematical bounds, not possible turn
              sequences. On a full board, changing an opponent stone to your color can only merge
              your groups or split theirs, so it cannot make your final score worse. If your
              opponent still wins after every open node is assigned to you, they have clinched
              the game — although you may keep playing.
            </p>
          </div>
          <p>
            The live score and territory colors are only a current projection and are not
            monotone. Creating a new separate star can lower its player&apos;s current star award,
            and projected territory can change owners, even though the completion bounds and
            crossed-out groups remain valid.
          </p>
          <p>
            The game ends only when the board is full.
          </p>
        </div>
      </article>
    </ModalDialog>
  );
}
