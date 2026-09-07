# Largest-board priority rollout — September 7, 2026

The user changed the objective to prioritize the largest board, selecting 85% of
training on ring 10 and 5% on each of rings 4, 6, and 8. All six game modes remain
equally important. Smaller-board regressions must not veto promotion when the
model improves on the largest board.

## Behavior

- GPU self-play chooses cohorts with weights 5/5/5/85. The learner applies the
  same proportions to homogeneous 512-position batches: 50/50/50/850 batches in
  a full 1,000-batch window. The existing CPU small-board actor supplies replay
  without overriding learner quotas.
- Promotion compares the six modes on ring 10. Existing statistical improvement
  tests and ring-10 mode regression guards remain active. Smaller boards are
  outside this promotion contract and cannot block it.
- Balanced evaluation uses fixed round targets. Interrupted sessions fill the
  current round's missing pairs before extending any board or mode. Previously,
  the next session repeatedly extended smaller boards toward the 40-pair cap,
  preventing completion of the initial full-board comparison.
- Historical strength measurement and headline Elo use the active ring-10
  contract. Reports verify the active profile checksum. Old 24-cell measurements
  and pending comparisons remain retained for diagnostics and cannot enter the
  new six-cell headline or promotion decision.
- The 17,402,775-parameter model, optimizer, EMA, UTD 1.5, all mode rules, 256-search
  screening, 1,024-search strength measurement, and cooperative GPU handoffs are
  unchanged. Changing board frequency does not increase the maximum batch size.

## Migration

The source is the cooperative release `6a8cee7` under
`/home/ubuntu/edgeconnect-releases/variant-cooperative-evaluation-20260905`.
The new release is built separately at
`/home/ubuntu/edgeconnect-releases/variant-ring10-priority-20260907` while training
continues. Deployment evidence is stored under
`/home/ubuntu/edgeconnect-rollouts/ring10-priority-20260907`.

The profile changes exactly five fields: the explicit training objective, ring
weight schedule, default self-play ring, arena ring set, and legacy regression-ring
list. The continuous migrator accepts a stopped nonterminal evaluation only
because the new balanced cell set has a distinct immutable contract. It pins and
backs up every old arena result and resume sidecar, rechecks their contents and
membership before writing, and records the objective transition in the migration
journal. Model, optimizer, checkpoint, learner, and UTD state remain intact.

The old incomplete comparison does not continue under the new objective. The
new contract selects the latest actionable candidate against the retained
champion. Earlier evaluation files stay unchanged in their original namespaces.

After a checkpointed stop, apply the validated migration, switch immutable
release/profile paths in the training and support services, and verify actual
85/5 learner sampling, ring-10 six-mode evaluation, automatic handoffs, and current
backups. Keep the old release and saved profile authority available for rollback.
