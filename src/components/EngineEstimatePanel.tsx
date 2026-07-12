import { useId, useMemo } from 'react';
import type { Board } from '@/lib/star/board';
import type { StarAiAnalysis } from '@/lib/star/ai/decision';

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatValue(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

function formatLatency(milliseconds: number): string {
  return milliseconds < 1_000
    ? `${milliseconds.toFixed(milliseconds < 10 ? 1 : 0)} ms`
    : `${(milliseconds / 1_000).toFixed(2)} s`;
}

function expectedMarginText(
  analysis: StarAiAnalysis,
  playerNames: readonly [string, string],
): string {
  if (Math.abs(analysis.expectedMargin) < 0.05) return 'Even (0.0 points)';
  const leader =
    analysis.expectedMargin > 0
      ? analysis.perspective
      : ((1 - analysis.perspective) as 0 | 1);
  return `${playerNames[leader]} +${Math.abs(analysis.expectedMargin).toFixed(1)} points`;
}

export interface EngineEstimatePanelProps {
  analysis: StarAiAnalysis;
  board: Board;
  playerNames: readonly [string, string];
}

export function EngineEstimatePanel({
  analysis,
  board,
  playerNames,
}: EngineEstimatePanelProps) {
  const titleId = useId();
  const winEstimates: [number, number] =
    analysis.perspective === 0
      ? [analysis.outcome.win, analysis.outcome.loss]
      : [analysis.outcome.loss, analysis.outcome.win];
  const candidates = useMemo(
    () =>
      analysis.rootActions
        .map((action, index) => ({
          action,
          visits: analysis.rootVisits[index],
          value: analysis.rootQ[index],
          index,
        }))
        .sort((left, right) => right.visits - left.visits || left.index - right.index)
        .slice(0, 3),
    [analysis],
  );

  return (
    <section
      role="status"
      aria-live="polite"
      aria-labelledby={titleId}
      className="rounded-2xl border border-gold/30 bg-gold-faint"
    >
      <details className="group px-4 py-2">
        <summary className="flex min-h-11 cursor-pointer list-none items-center text-sm text-gold-strong marker:text-gold">
          <span id={titleId} className="font-medium">
            Engine estimate
          </span>
          <span className="ml-2 truncate text-xs text-muted">
            from {playerNames[analysis.perspective]}&apos;s turn
          </span>
        </summary>

        <div className="mb-2 mt-1 grid gap-2 text-xs sm:grid-cols-2">
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Win estimates
            </p>
            <p className="mt-1 text-ink">
              {playerNames[0]} {formatPercent(winEstimates[0])}
            </p>
            <p className="text-ink">
              {playerNames[1]} {formatPercent(winEstimates[1])}
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Expected margin
            </p>
            <p className="mt-1 text-ink">
              {expectedMarginText(analysis, playerNames)}
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Search value
            </p>
            <p className="mt-1 font-mono text-ink">
              {formatValue(analysis.searchValue)}
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Simulations
            </p>
            <p className="mt-1 text-ink">
              {analysis.simulations.toLocaleString('en-US')}
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Latency
            </p>
            <p className="mt-1 text-ink">
              {formatLatency(analysis.timingMs.total)}
            </p>
          </div>
          <div className="rounded-lg border border-white/10 bg-black/10 px-3 py-2">
            <p className="text-xs uppercase tracking-[0.12em] text-muted">
              Model step
            </p>
            <p className="mt-1 text-ink">
              {analysis.modelStep === null
                ? 'Not reported'
                : analysis.modelStep.toLocaleString('en-US')}
            </p>
          </div>
        </div>

        <div className="mt-3">
          <p className="text-xs uppercase tracking-[0.12em] text-muted">
            Top candidate moves by visits
          </p>
          <ol className="mt-1.5 space-y-1">
            {candidates.map((candidate) => (
              <li
                key={candidate.action.node}
                className="grid grid-cols-[auto_1fr] items-baseline gap-2 rounded-lg bg-black/10 px-2.5 py-1.5 text-xs sm:grid-cols-[auto_1fr_auto]"
              >
                <span className="font-mono text-gold-strong">
                  {board.labels[candidate.action.node]}
                </span>
                <span className="text-muted">
                  {candidate.visits.toLocaleString('en-US')} visits
                </span>
                <span className="col-span-2 font-mono text-ink sm:col-span-1">
                  value {formatValue(candidate.value)}
                </span>
              </li>
            ))}
          </ol>
        </div>

        <p className="sr-only">
          Analysis for {analysis.stateHash}, player{' '}
          {analysis.perspective + 1} perspective.
        </p>
      </details>
    </section>
  );
}
