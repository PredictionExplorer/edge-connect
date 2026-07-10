'use client';

import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

interface RulesDialogProps {
  open: boolean;
  onClose: () => void;
}

export function RulesDialog({ open, onClose }: RulesDialogProps) {
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [onClose, open]);
  if (!open) return null;
  return (
    <div
      className="fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal
      aria-label="How to play *Star"
    >
      <article
        onClick={(e) => e.stopPropagation()}
        className="fade-up thin-scroll relative max-h-[85dvh] w-full max-w-2xl overflow-y-auto rounded-3xl border border-gold/30 bg-night-raise p-8 shadow-2xl"
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

        <h2 className="font-display text-3xl text-gold-strong">How to play</h2>

        <div className="mt-4 space-y-4 text-sm leading-relaxed text-ink/90">
          <p>
            Two players take turns placing stones on empty nodes — one stone per turn in{' '}
            <em>*Star</em>, two per turn in <em>Double *Star</em> (the first player places just
            one stone on the game&apos;s very first turn). Stones never move. The star-shaped{' '}
            <strong className="text-gold">bridge</strong> in the center cannot be played, but it
            connects all five innermost nodes for <em>both</em> players.
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
            everything; waste nothing. Ties in points go to whoever holds more quarks, and on a
            finished board the two totals always sum to the number of peries plus one — so
            someone always wins.
          </p>
          <p>
            The game ends when the board is full or when both players pass in succession,
            agreeing nothing more can change.
          </p>
        </div>
      </article>
    </div>
  );
}
