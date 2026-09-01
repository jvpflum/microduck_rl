# V11 control × iteration-6159 speed hybrid

This directory preserves the first official-friction MicroDuck candidate that
combines the all-around V11 control policy with the iteration-6159 speed
specialist in one deployable ONNX graph.

The graph uses V11 for cruise, stopping, and turning. For a forward command
above `0.5 m/s` with absolute yaw command below `0.25 rad/s`, it emits a
`90%` iteration-6159 / `10%` V11 action blend. The policy remains a standard
`[1, 61] -> [1, 14]` ONNX actor and requires no custom runtime.

## Official Race5 result

- Wheel frictionloss: `0.003`
- Current limit: `1.75 A`
- Qualification gates: `14 / 14 passed`
- 100-foot time: `26.88498 s`
- Course-average speed: `2.53606 mph`
- Trap speed: `2.83357 mph`
- Race sustained speed: `2.098 mph`
- Verified 0.5-second top speed: `2.92584 mph`
- Instantaneous world-X peak: `3.25499 mph`
- Maximum long-run drift: `1.39 ft`
- Maximum long-run heading error: `16.0 deg`
- Full V11 cruise, brake, left-turn, right-turn, and stability circuit: passed

The five-rollout 20-second perturbation check survived `5 / 5` with no falls,
`2.36408 mph` mean body-forward speed, and `2.35878 mph` mean world-X speed.

## Immutable hashes

- `hybrid_v11_control_i6159_speed_blend90.onnx`:
  `eb637d92c2ca2854c13ed60824b26fe9472809a29b6e0b76c615aba476ca3a3f`
- `hybrid_v11_control_i6159_speed_blend90_race5.json`:
  `8ffd0ee1db68ba4387065c4f0d735ba591c2ef1e169478ea960e123a0b312802`
- `hybrid_v11_control_i6159_speed_blend90_robustness.json`:
  `128b3b113f44e7090e4023e45923753aade49555d7d2f64e82f5a2077d0edefe`
