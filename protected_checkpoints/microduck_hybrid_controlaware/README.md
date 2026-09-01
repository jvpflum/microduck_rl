# Control-aware V11 × iteration-6159 hybrid

This candidate fixes the original hybrid's hard routing discontinuity. At a
forward command above `0.5 m/s`, the speed specialist receives full authority
while absolute yaw correction is at or below `0.02 rad/s`; its authority then
decreases linearly to zero at `0.12 rad/s`. V11 owns cruise, braking, turning,
and larger straight-line corrections. The exported graph remains a standard
`[1, 61] -> [1, 14]` ONNX actor and needs no custom policy runtime.

The official Race5 replay uses wheel frictionloss `0.003`, current limit
`1.75 A`, and bounded line hold with `yaw_kp=0.55`, `lateral_kp=0.25`,
`yaw_kd=0.05`, and `max_correction=0.10 rad/s`.

## Official Race5 result

- Qualification gates: `14 / 14 passed`
- Pollen head-to-head dimensions: `9 / 9 improved`
- 100-foot time: `29.47022 s` (Pollen `57.58946 s`)
- Course-average speed: `2.31358 mph`
- Sustained speed: `2.009 mph` (Pollen `1.066 mph`)
- Verified 0.5-second top speed: `2.662 mph` (Pollen `1.283 mph`)
- Trap speed: `2.53360 mph` (Pollen `1.220 mph`)
- First-second acceleration: `0.991 mph/s` (Pollen `0.723 mph/s`)
- Time to `0.5 m/s`: `0.90 s`
- Maximum long-run drift: `0.96 ft` (Pollen `1.25 ft`)
- Maximum long-run heading error: `9.51 deg` (Pollen `11.06 deg`)
- Automatic steering: `14.8%` (Pollen `16.4%`)
- Retained agility: `67.77 / 100` (Pollen `56.59 / 100`)

The five-rollout perturbation battery survived `5 / 5` with no falls,
`2.20322 mph` sustained world-X speed, `2.62 deg` mean absolute heading, and
`0.22954 m` mean maximum lateral deviation.

## Immutable hashes

- `hybrid_v11_i6159_smooth_t02_t12_b100.onnx`:
  `b736f186f30f73d38030b7fad57f682bd014ce813bdc5f1b5698f3efb7b7be6a`
- `hybrid_v11_i6159_smooth_t02_t12_b100_race5.json`:
  `adde6c39217952bbd110d0657b3f3ef7ea7ca41659d6a460b1f7c7c9ad5baca5`
- `hybrid_v11_i6159_smooth_t02_t12_b100_robustness.json`:
  `c37f4a105cca6af7f0c76fb950d62cb605a0f53a620b26c9fd83c3feaeaaf420`
