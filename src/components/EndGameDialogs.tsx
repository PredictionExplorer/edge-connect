'use client';

import { useId, useRef, type RefObject } from 'react';
import { Eye, Flag, ShieldCheck, Trophy } from 'lucide-react';
import { ModalDialog } from './ModalDialog';
import { PLAYER_COLORS } from './theme';

interface ClinchDialogProps {
  open: boolean;
  winner: 0 | 1;
  winnerName: string;
  loserName: string;
  emptyNodes: number;
  proofScores: readonly [number, number];
  returnFocusRef?: RefObject<HTMLElement | null>;
  onContinue: () => void;
  onProof: () => void;
  onEnd: () => void;
}

export function ClinchDialog({
  open,
  winner,
  winnerName,
  loserName,
  emptyNodes,
  proofScores,
  returnFocusRef,
  onContinue,
  onProof,
  onEnd,
}: ClinchDialogProps) {
  const titleId = useId();
  const descriptionId = useId();
  const continueButton = useRef<HTMLButtonElement>(null);
  const openLabel = `${emptyNodes} open node${emptyNodes === 1 ? '' : 's'}`;

  return (
    <ModalDialog
      open={open}
      onClose={onContinue}
      labelledBy={titleId}
      describedBy={descriptionId}
      initialFocusRef={continueButton}
      returnFocusRef={returnFocusRef}
      closeOnBackdrop={false}
      className="max-w-xl"
    >
      <div className="thin-scroll panel-surface max-h-[calc(100dvh-1rem)] w-full overflow-y-auto rounded-[1.75rem] p-5 text-center shadow-[0_0_90px_rgba(232,196,139,0.2)] sm:p-8">
        <div className="pop-in mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-gold/50 bg-gold-faint">
          <ShieldCheck className="h-7 w-7 text-gold-strong" aria-hidden />
        </div>
        <p className="fade-in text-xs font-semibold uppercase tracking-[0.18em] text-gold">
          Result clinched
        </p>
        <h2 id={titleId} className="font-display fade-up mt-1 text-3xl text-ink sm:text-4xl">
          <span style={{ color: PLAYER_COLORS[winner].bright }}>{winnerName}</span>{' '}
          cannot be caught
        </h2>
        <p
          id={descriptionId}
          className="fade-in mx-auto mt-3 max-w-md text-sm leading-relaxed text-muted"
        >
          Even if every remaining open node became {loserName}&apos;s, {winnerName}{' '}
          would still win.
        </p>

        <div className="fade-up mx-auto mt-5 max-w-md rounded-2xl border border-gold/25 bg-gold-faint px-4 py-3 text-left">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-semibold uppercase tracking-[0.14em] text-gold">
              Strongest-case proof
            </span>
            <span className="shrink-0 text-xs tabular-nums text-muted">{openLabel}</span>
          </div>
          <p className="mt-2 text-sm text-ink">
            All open nodes → {loserName}
          </p>
          <p className="mt-0.5 text-xs leading-relaxed text-muted">
            Hypothetical boundary: {winnerName} {proofScores[winner]} · {loserName}{' '}
            {proofScores[1 - winner]}
          </p>
        </div>

        <p className="mt-4 text-xs leading-relaxed text-muted">
          Ending now records the winner without inventing a final score.
        </p>

        <div className="sticky bottom-0 z-10 -mx-2 mt-5 grid grid-cols-2 gap-2 rounded-2xl bg-night-surface-strong/95 px-2 py-3 backdrop-blur-md sm:grid-cols-3">
          <button
            ref={continueButton}
            type="button"
            onClick={onContinue}
            className="flex min-h-11 items-center justify-center rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors hover:border-gold/50"
          >
            Continue playing
          </button>
          <button
            type="button"
            onClick={onProof}
            className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/20 px-3 py-2 text-sm text-ink transition-colors hover:border-gold/50 hover:bg-white/[0.04]"
          >
            <Eye className="h-4 w-4" aria-hidden /> Show proof board
          </button>
          <button
            type="button"
            onClick={onEnd}
            className="col-span-2 flex min-h-11 items-center justify-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-3 py-2 text-sm font-medium text-gold-strong transition-colors hover:bg-gold/25 sm:col-span-1"
          >
            <Trophy className="h-4 w-4" aria-hidden /> End game now
          </button>
        </div>
      </div>
    </ModalDialog>
  );
}

interface ConfirmationDialogProps {
  open: boolean;
  eyebrow: string;
  title: string;
  description: string;
  confirmLabel: string;
  destructive?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

function ConfirmationDialog({
  open,
  eyebrow,
  title,
  description,
  confirmLabel,
  destructive = false,
  onCancel,
  onConfirm,
}: ConfirmationDialogProps) {
  const titleId = useId();
  const descriptionId = useId();
  const cancelButton = useRef<HTMLButtonElement>(null);

  return (
    <ModalDialog
      open={open}
      onClose={onCancel}
      labelledBy={titleId}
      describedBy={descriptionId}
      initialFocusRef={cancelButton}
      closeOnBackdrop={false}
      className="max-w-md"
    >
      <div className="panel-surface w-full rounded-3xl p-5 text-center shadow-[0_0_70px_rgba(0,0,0,0.35)] sm:p-7">
        <div
          className={`mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full border ${
            destructive
              ? 'border-danger/45 bg-danger/[0.08] text-danger'
              : 'border-gold/45 bg-gold-faint text-gold-strong'
          }`}
        >
          <Flag className="h-5 w-5" aria-hidden />
        </div>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted">
          {eyebrow}
        </p>
        <h2 id={titleId} className="font-display mt-1 text-3xl text-ink">
          {title}
        </h2>
        <p id={descriptionId} className="mt-3 text-sm leading-relaxed text-muted">
          {description}
        </p>
        <div className="mt-6 grid gap-2 sm:grid-cols-2">
          <button
            ref={cancelButton}
            type="button"
            onClick={onCancel}
            className="min-h-11 rounded-xl border border-white/15 px-4 py-2 text-sm text-ink transition-colors hover:border-gold/50"
          >
            Keep playing
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`min-h-11 rounded-xl border px-4 py-2 text-sm font-medium transition-colors ${
              destructive
                ? 'border-danger/60 bg-danger/[0.08] text-danger hover:bg-danger/[0.16]'
                : 'border-gold/60 bg-gold-faint text-gold-strong hover:bg-gold/25'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </ModalDialog>
  );
}

interface EndGameConfirmDialogProps {
  open: boolean;
  winnerName: string;
  emptyNodes: number;
  onCancel: () => void;
  onConfirm: () => void;
}

export function EndGameConfirmDialog({
  open,
  winnerName,
  emptyNodes,
  onCancel,
  onConfirm,
}: EndGameConfirmDialogProps) {
  return (
    <ConfirmationDialog
      open={open}
      eyebrow="Result clinched"
      title="End this clinched game?"
      description={`${winnerName} is guaranteed to win. Ending now leaves ${emptyNodes} open node${
        emptyNodes === 1 ? '' : 's'
      } and records no final score.`}
      confirmLabel="End game"
      onCancel={onCancel}
      onConfirm={onConfirm}
    />
  );
}

interface ResignDialogProps {
  open: boolean;
  loserName: string;
  winnerName: string;
  onCancel: () => void;
  onConfirm: () => void;
}

export function ResignDialog({
  open,
  loserName,
  winnerName,
  onCancel,
  onConfirm,
}: ResignDialogProps) {
  return (
    <ConfirmationDialog
      open={open}
      eyebrow="Resignation"
      title={`Resign ${loserName}?`}
      description={`${winnerName} will win immediately. This game will end without a final score.`}
      confirmLabel={`Resign ${loserName}`}
      destructive
      onCancel={onCancel}
      onConfirm={onConfirm}
    />
  );
}
