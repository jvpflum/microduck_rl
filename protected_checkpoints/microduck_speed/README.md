# Protected MicroDuck speed checkpoints

These artifacts are intentionally versioned even though generated models are
normally ignored.  They are the immutable recovery points before the
official-friction frontier experiments.

## Official-friction champion

- Checkpoint: `official_friction_champion_model_6159.pt`
- Deployment actor: `official_friction_champion.onnx`
- Evaluation: `official_friction_champion_eval.json`
- Wheel bearing frictionloss: `0.003 N m` per passive wheel
- Sustained world-X speed: `1.0991368399 m/s` (`2.458699 mph`)
- Survival: `1.0`

## Frictionless speed donor

- Checkpoint: `frictionless_speed_donor_model_160.pt`
- Deployment actor: `frictionless_speed_donor.onnx`
- Evaluation: `frictionless_speed_donor_eval.json`
- Wheel bearing frictionloss: `0.0 N m`
- Sustained world-X speed: `1.8778962528 m/s` (`4.200734 mph`)
- Maximum observed speed: `2.4162330064 m/s` (`5.404959 mph`)
- Survival: `1.0`

## SHA-256

```text
ff3e5f0efc15ea91cdb09528ba8aee90a531583ff9a9548de35c6c128131d2ab  frictionless_speed_donor.onnx
0b62cd78a785115c07fd1599118effd1d1e9ac21fea85fc8380a09c187ae05ba  frictionless_speed_donor_eval.json
89788f0ac9fcb9814e24f92a41ff9728e341341a963017d0794b8d89d5a2beb0  frictionless_speed_donor_model_160.pt
1546522c8c9806e4a3a23538de4fe86a9afc85cfe4a3086f8bf392d3bbd9b900  official_friction_champion.onnx
aa28150adaf0bbb0281b0385f5e842d6b535b977d1ed8d34ec149bf6b7b4320a  official_friction_champion_eval.json
50152ba1d27931d2f77c6a5be78b12310b1caadbc54123ff331684bf03c09e24  official_friction_champion_model_6159.pt
```
