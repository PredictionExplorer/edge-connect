'use client';

import { useEffect, useId, useState } from 'react';
import type {
  AiCapabilities,
  AiSearchCapability,
} from '@/lib/star/ai/capabilities';
import type { StarAiSearchBudget } from '@/lib/star/ai/decision';
import { useAppStore, type AiRuntime } from '@/lib/store';
import { engineControllerLabel } from './starAiDevtools';

const PRESETS = [
  ['quick', 'Quick'],
  ['strong', 'Strong'],
  ['maximum', 'Maximum'],
] as const;

function parseBudgetInteger(value: string, maximum: number): number | null {
  if (!/^[1-9][0-9]*$/.test(value)) return null;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed <= maximum ? parsed : null;
}

function budgetError(value: string, maximum: number, label: string): string | null {
  return parseBudgetInteger(value, maximum) === null
    ? `${label} must be a whole number from 1 to ${maximum.toLocaleString('en-US')}.`
    : null;
}

export function budgetFitsCapability(
  budget: StarAiSearchBudget,
  search: AiSearchCapability | undefined,
): boolean {
  return (
    search === undefined ||
    (Number.isSafeInteger(budget.simulations) &&
      budget.simulations > 0 &&
      budget.simulations <= search.maximum.simulations &&
      Number.isSafeInteger(budget.maxConsidered) &&
      budget.maxConsidered > 0 &&
      budget.maxConsidered <= search.maximum.maxConsidered)
  );
}

interface EngineBudgetControlsProps {
  runtime: AiRuntime;
  search: AiSearchCapability;
  budget: StarAiSearchBudget;
  onChange: (budget: StarAiSearchBudget) => void;
  onValidityChange: (runtime: AiRuntime, valid: boolean) => void;
}

function EngineBudgetControls({
  runtime,
  search,
  budget,
  onChange,
  onValidityChange,
}: EngineBudgetControlsProps) {
  const [draft, setDraft] = useState({
    simulations: String(budget.simulations),
    maxConsidered: String(budget.maxConsidered),
  });
  const inputId = useId();
  const simulationsError = budgetError(
    draft.simulations,
    search.maximum.simulations,
    'Simulations',
  );
  const maxConsideredError = budgetError(
    draft.maxConsidered,
    search.maximum.maxConsidered,
    'Max considered',
  );
  const valid = simulationsError === null && maxConsideredError === null;

  useEffect(() => {
    onValidityChange(runtime, valid);
  }, [onValidityChange, runtime, valid]);

  const applyBudget = (next: StarAiSearchBudget) => {
    setDraft({
      simulations: String(next.simulations),
      maxConsidered: String(next.maxConsidered),
    });
    onChange(next);
  };

  const updateDraft = (
    field: keyof StarAiSearchBudget,
    value: string,
  ) => {
    const next = { ...draft, [field]: value };
    setDraft(next);
    const simulations = parseBudgetInteger(
      next.simulations,
      search.maximum.simulations,
    );
    const maxConsidered = parseBudgetInteger(
      next.maxConsidered,
      search.maximum.maxConsidered,
    );
    if (simulations !== null && maxConsidered !== null) {
      onChange({ simulations, maxConsidered });
    }
  };

  return (
    <>
      <div
        role="group"
        aria-label={`${engineControllerLabel(runtime)} effort preset`}
        className="grid gap-2 sm:grid-cols-3"
      >
        {PRESETS.flatMap(([key, label]) => {
          const preset = search.presets[key];
          if (!preset) return [];
          const selected =
            draft.simulations === String(preset.simulations) &&
            draft.maxConsidered === String(preset.maxConsidered);
          return [
            <button
              key={key}
              type="button"
              aria-pressed={selected}
              aria-label={`${label}, ${preset.simulations.toLocaleString(
                'en-US',
              )} simulations, up to ${preset.maxConsidered.toLocaleString(
                'en-US',
              )} candidates`}
              onClick={() => applyBudget({ ...preset })}
              className={`rounded-xl border px-3 py-2 text-left transition-colors ${
                selected
                  ? 'border-gold/70 bg-gold-faint'
                  : 'border-white/10 bg-white/[0.03] hover:border-gold/35'
              }`}
            >
              <span className="block text-sm text-ink">{label}</span>
              <span className="mt-0.5 block text-[10px] leading-relaxed text-muted">
                {preset.simulations.toLocaleString('en-US')} simulations · up to{' '}
                {preset.maxConsidered.toLocaleString('en-US')} candidates
              </span>
            </button>,
          ];
        })}
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        Runs exactly {budget.simulations.toLocaleString('en-US')} simulations and
        considers up to {budget.maxConsidered.toLocaleString('en-US')} moves.
      </p>

      <details className="mt-3 rounded-lg border border-white/10 bg-black/10 px-3 py-2">
        <summary className="cursor-pointer text-xs text-gold-strong">
          Advanced search budget
        </summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <div>
            <label
              htmlFor={`${inputId}-simulations`}
              className="block text-xs text-ink"
            >
              Simulations
            </label>
            <input
              id={`${inputId}-simulations`}
              type="number"
              inputMode="numeric"
              min={1}
              max={search.maximum.simulations}
              step={1}
              value={draft.simulations}
              aria-invalid={simulationsError !== null}
              aria-describedby={`${inputId}-simulations-hint${
                simulationsError ? ` ${inputId}-simulations-error` : ''
              }`}
              onChange={(event) =>
                updateDraft('simulations', event.target.value)
              }
              className="mt-1 w-full rounded-lg border border-white/15 bg-white/[0.04] px-2.5 py-2 text-sm text-ink outline-none focus:border-gold/60"
            />
            <p
              id={`${inputId}-simulations-hint`}
              className="mt-1 text-[10px] text-muted"
            >
              Exact count, maximum{' '}
              {search.maximum.simulations.toLocaleString('en-US')}.
            </p>
            {simulationsError && (
              <p
                id={`${inputId}-simulations-error`}
                role="alert"
                className="mt-1 text-[10px] text-danger"
              >
                {simulationsError}
              </p>
            )}
          </div>

          <div>
            <label
              htmlFor={`${inputId}-max-considered`}
              className="block text-xs text-ink"
            >
              Max considered
            </label>
            <input
              id={`${inputId}-max-considered`}
              type="number"
              inputMode="numeric"
              min={1}
              max={search.maximum.maxConsidered}
              step={1}
              value={draft.maxConsidered}
              aria-invalid={maxConsideredError !== null}
              aria-describedby={`${inputId}-max-considered-hint${
                maxConsideredError ? ` ${inputId}-max-considered-error` : ''
              }`}
              onChange={(event) =>
                updateDraft('maxConsidered', event.target.value)
              }
              className="mt-1 w-full rounded-lg border border-white/15 bg-white/[0.04] px-2.5 py-2 text-sm text-ink outline-none focus:border-gold/60"
            />
            <p
              id={`${inputId}-max-considered-hint`}
              className="mt-1 text-[10px] text-muted"
            >
              Root candidates, maximum{' '}
              {search.maximum.maxConsidered.toLocaleString('en-US')}.
            </p>
            {maxConsideredError && (
              <p
                id={`${inputId}-max-considered-error`}
                role="alert"
                className="mt-1 text-[10px] text-danger"
              >
                {maxConsideredError}
              </p>
            )}
          </div>
        </div>
      </details>
    </>
  );
}

export interface EngineDeveloperSettingsProps {
  runtimes: readonly AiRuntime[];
  capabilities: AiCapabilities;
  onValidityChange: (runtime: AiRuntime, valid: boolean) => void;
}

export function EngineDeveloperSettings({
  runtimes,
  capabilities,
  onValidityChange,
}: EngineDeveloperSettingsProps) {
  const settings = useAppStore((state) => state.aiSearchSettings);
  const setAiSearchBudget = useAppStore((state) => state.setAiSearchBudget);

  return (
    <details className="rounded-xl border border-gold/25 bg-gold-faint px-4 py-3">
      <summary className="cursor-pointer text-sm font-medium text-gold-strong">
        Engine developer settings
      </summary>
      <div className="mt-4 space-y-4">
        {runtimes.map((runtime) => {
          const capability = capabilities[runtime];
          const search =
            capability.status === 'available' ? capability.search : undefined;
          return (
            <fieldset
              key={runtime}
              className="rounded-xl border border-white/10 bg-white/[0.025] p-3"
            >
              <legend className="px-1 text-xs font-medium text-ink">
                {engineControllerLabel(runtime)}
              </legend>
              {search ? (
                <EngineBudgetControls
                  runtime={runtime}
                  search={search}
                  budget={settings[runtime]}
                  onChange={(budget) => setAiSearchBudget(runtime, budget)}
                  onValidityChange={onValidityChange}
                />
              ) : (
                <p role="status" className="text-[11px] leading-relaxed text-muted">
                  Search metadata is not available for this engine.
                </p>
              )}
            </fieldset>
          );
        })}
      </div>
    </details>
  );
}
