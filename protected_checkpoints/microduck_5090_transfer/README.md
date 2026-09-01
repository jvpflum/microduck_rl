# DuckLab 5090 review bundle — 2026-09-01

This bundle intentionally contains only deployable ONNX policies, scrubbed
native-MuJoCo evaluation JSON, recipes, and hashes. It contains no raw `.pt`
checkpoint, optimizer state, training log, W&B data, or secret.

## `v35-speed-linehold`

Preserved pre-calibration speed/line-hold candidate. It is a useful control, not
an all-around drive policy: at a zero command it moves and it does not brake or
turn correctly.

- Policy SHA-256: `8108992d8d8b2096bbf503404e3cf40c06aa41664c8a34db1cc3265eb8262b5e`
- Official wheel friction used by the supplied native evaluations: `0.003`

## `v47-speed-specialist`

Best current standalone high-speed specialist. Native official-friction test:
1.204 m/s steady max-speed segment, 100 ft in 26.145 s, 0.521 m maximum lateral
drift, and 18.2 degrees maximum heading error. It is not a complete drive
controller: zero-command, braking, and turning require the multi-skill router.

- Policy SHA-256: `6079db680499a771ef34a9d391b97eee4276332df54bd3a461a7362e021add87`
- Official wheel friction: `0.003`
- Official evaluator current limit: `1.75 A`

## `v53-multiskill-experimental`

Command-gated composite of the V47 speed specialist, the V35 low-speed straight
specialist, and Pollen's `BEST_roller` stand/turn controller. It idles, turns
about +68.5/-70.0 degrees in four seconds, retains V47's exact 100-ft result,
and reduces six-second low-speed yaw to about 13 degrees. It is experimental and
must not be promoted as champion yet: full-speed braking stays upright but still
slides laterally and needs the V55 training run.

- Policy SHA-256: `380c2a4d2353dc26a9efa51f384257056c3decbd169351c997bf1e8ecfa436ae`
- Required preview/controller deceleration ramp: `0.30 m/s^2` for the initial
  safe-braking configuration; native sweeps are included for review.

Promotion rule: compare every candidate against the Pollen Race5 baseline and
the preserved V47 speed specialist using native MuJoCo at friction `0.003` and
current limit `1.75 A`. Do not promote on PPO reward alone.
