# Resumable evaluation scheduling — September 5, 2026

The balanced historical measurement of champion 19,532 against predecessor 9,766
held GPU 7 for approximately 13 hours while candidate 53,713 waited. Handicap
searches can take much longer than a normal wave, so a wave-count limit alone did
not bound GPU occupancy.

## Scheduling and persistence

- Ready, nonterminal promotion candidates precede historical measurements.
  Started candidates retain their existing finish-inflight policy, including
  candidates whose first paired result is not complete yet.
- Balanced production evaluations get 300 seconds per GPU lease. Stage B
  promotion leases have a 300-second self-play cooldown; historical leases have
  an independent 1,800-second cooldown. Historical cooldown never blocks a new
  candidate, and an arriving candidate interrupts historical work within the
  polling interval (at most 10 seconds, plus the current native/inference call).
- Atomic `.resume.json` sidecars retain completed moves and finished games,
  including a finished seat whose partner is still running. A restarted search
  repeats only its unfinished move. Evaluation statistics include complete pairs
  only. Resume contracts bind model identities, rules, search budgets, openings,
  and seed policy; invalid or incompatible snapshots fail closed.
- Unfinished predecessor measurements remain eligible after a newer champion is
  promoted. Historical work resumes during idle candidate periods.
- `evaluation-session-events.jsonl` records lease duration, stop reason, and
  durable move/game/pair progress. Scheduling state survives coordinator restarts.

Move-level resumability and wall-time slicing apply to the balanced arena, which
already uses stable per-game seeds. Legacy nonbalanced evaluations retain their
existing seed policy, completed-pair persistence, and wave lease limits to avoid
mixing incompatible evidence or repeatedly abandoning long games.

This change preserves the 17,402,775-parameter model, optimizer, learner/replay
settings, all six game modes, equal board allocation, and the 256-search promotion
and 1,024-search strength contracts. It changes when evaluation runs, not its
statistical requirements. No Elo-per-hour improvement is claimed before measured
strength results are available.

## Deployment procedure

Build and validate a new release at
`/home/ubuntu/edgeconnect-releases/variant-evaluation-sessions-20260905` while the
active runtime continues. Save units, profile, recovery provenance, and backups
under `/home/ubuntu/edgeconnect-rollouts/evaluation-sessions-20260905` before a
graceful stop. The old runtime can retain completed pairs at shutdown but cannot
retroactively save moves from a game begun before this fix.

Apply the continuous-profile migrator to a clone of the active profile with
explicit session defaults and the Stage B candidate cooldown reduced from 1,800
to 300 seconds. The migrator must confirm that training state and evaluation
contracts remain unchanged. Pin source and native hashes, freeze the new release,
update the training and support units, then restart from the saved checkpoint.
Verify candidate priority, advancing learner steps, a bounded lease, durable game
progress, actor recovery during cooldown, and current backups.

The previous release, profile, and unit files remain available for rollback.
