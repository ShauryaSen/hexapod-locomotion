# How the hexapod learned to walk, and what changed in the code

Written for: Shaurya, who reviewed the pre-overnight version of this package line by line.
Labels: **measured** = a number from a run; **code** = read from source; **remembered** = general knowledge.

## 1. The method in one paragraph

Six PPO runs from a standing start never discovered a coordinated six-leg sweep: the noisy policy earned velocity
rewards from random twitching while the noise-free policy stood still (measured; `logs/overnight/README.md`).
So instead of asking PPO to *discover* the gait, we give the network the gait first and let PPO *improve* it:

1. **Teacher.** The club's scripted tripod, written as joint-angle offsets from the standing pose as a function of a
   clock phase and the commanded speed (`tools/bc_clone.py::ScriptedTripod`). It walks 0.09-0.13 m/s open-loop (measured).
2. **Behavior cloning (BC).** Run the teacher in the simulator and record pairs (what the policy observes -> what the
   teacher does). Fit the same MLP shape PPO uses (ELU 68-128-128-128-18) by least squares. Plain BC walked at
   0.039 m/s (measured): small errors push the robot into states the data never covered and they compound.
3. **DAgger.** Let the clone drive, label every state it reaches with the teacher (the teacher only needs the phase and
   the command, so any state can be labelled), add those pairs, refit. Two rounds: 0.039 -> 0.072 m/s (measured).
4. **Splice.** Copy the clone's weights into a checkpoint that rsl_rl itself wrote (so the critic and optimizer have
   the right layout), set the exploration std to 0.1, clear Adam's moments, save as `model_bc.pt`.
5. **PPO fine-tune.** Resume training from that checkpoint with a low learning rate (2e-4). The reward is the
   ordinary velocity-tracking reward; nothing scripted touches the actions. Result: 0.100 -> 0.145 m/s along the
   command and 48 -> 21 degrees of heading error over 600 iterations, zero falls (measured, run bc1).

The deployed policy is a plain MLP. The scripted gait was used as *training data* only.

## 2. Every change from the pre-overnight code (`git diff pre-overnight clean-bc`)

### `hexapod_env_cfg.py` (+13 lines)
| change | why |
|---|---|
| `viewer = ViewerCfg(eye=(0.45,-0.45,0.22), lookat=(0,0,0.1), origin_type="asset_root", asset_name="robot")` | videos were taken from 7.5 m away; the robot is 0.1 m tall. Camera now sits next to env 0 |
| `command_forward_only: bool = False`, `command_lin_vel_min: float = 0.0` | the teacher only walks forward/backward along body y, so training commands must match: `|v| ~ U(min, max)` along +-y, x = 0 |
| `gait_clock_enabled: bool = False`, `gait_period_s: float = 0.8` | the teacher is a function of a clock phase; the clone needs the same clock as an input (a memoryless MLP cannot keep time from a standing start) |

### `hexapod_env.py` (+33 lines)
| change | why |
|---|---|
| `import math` | for `2 * math.pi` |
| `__init__`: `cfg.observation_space = base_obs + (2 if cfg.gait_clock_enabled else 0)` before `super().__init__` | the observation grows by `[sin, cos]` of the phase; set here because Hydra command-line overrides are applied after the cfg class body runs |
| `__init__`: `_speed_xy_sum`, `_swing_sum`, `_torque_sat_sum`, `_torque_cap` | logging only: episode-mean body speed, fraction of steps with a foot in the air, fraction of joints at the torque cap. These are how a run is judged without a video |
| `__init__`: `_gait_phase` | one clock phase per env [rad] |
| `_pre_physics_step`: advance `_gait_phase` by `2*pi*step_dt/gait_period_s` when enabled | the clock ticks once per policy step (50 Hz); period 0.8 s = 40 steps per gait cycle |
| `_get_observations`: append `[sin(phase), cos(phase)]` when enabled | the two extra observation values |
| `_get_rewards`: accumulate the three metrics | logging only; no reward term changed |
| `_reset_idx`: `_gait_phase[env_ids] = rand * 2*pi` | episodes start at a random point of the cycle so the policy sees every phase |
| `_reset_idx`: forward-only command sampling branch | see cfg; the old uniform-square sampling is kept as the `else` |
| `_reset_idx`: flush/zero the three metrics | logging only |

Nothing in the reward function, terminations, robot config, or PPO config changed relative to the version you
reviewed. Every reward/curriculum/termination experiment from the overnight and from rl1-rl6/dm1 is gone from the
working files; their code is preserved under the git tags `overnight`, `rl1`, `rl2`, `rl4`, `rl5`, `rl6`, `dm1`.

### `tools/bc_clone.py` (new, 199 lines)
The teacher gait (`ScriptedTripod`), data collection with DAgger, the MLP fit, the standing-start verification, and
the checkpoint splice. Docstrings at the top of the file and of the class explain each piece.

## 3. How to reproduce (inside the container, from the Isaac Lab root)

```bash
OVR="env.command_forward_only=True env.command_lin_vel_min=0.10 env.command_lin_vel_max=0.15 env.command_yaw_rate_max=0.1 env.gait_clock_enabled=True env.feet_air_time_threshold=0.0 agent.algorithm.learning_rate=2.0e-4"
# 1. a 1-iteration run just to get a checkpoint with the right layout
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Hexapod-Flat-Direct-v0 --external_callback hexapod.register --headless --num_envs 512 --max_iterations 1 --run_name init $OVR
# 2. clone the teacher and splice it into that checkpoint (writes model_bc.pt next to model_0.pt)
./isaaclab.sh -p /workspace/learning/hexapod/tools/bc_clone.py --headless --splice_into /workspace/isaaclab/logs/rsl_rl/hexapod_flat_direct/<init run>/model_0.pt
# 3. fine-tune with PPO from the clone
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Hexapod-Flat-Direct-v0 --external_callback hexapod.register --headless --num_envs 512 --max_iterations 600 --resume --load_run <init run> --checkpoint model_bc.pt --run_name bc $OVR
```
(`PYTHONPATH=/workspace/learning:$PYTHONPATH` must be set, as for every hexapod command.)
`feet_air_time_threshold=0.0` means any touchdown pays `0.5 * swing time`; the pre-overnight default 0.2 s penalized
the teacher's ~0.3 s swings only slightly, but 0 is what the verified runs used.

## 3b. Verification of the clean code (run bcmin, 2026-10-14, measured)
Noise-free rollouts from a standing start, 32 robots, commands 0.10-0.15 m/s:

| checkpoint | speed along command | heading error | falls |
|---|---|---|---|
| clone (model_0) | 0.102 m/s | 46 deg | 0 |
| model_100 | 0.106 m/s | 8.8 deg | 0 |
| model_200 | 0.121 m/s | 5.4 deg | 0 |
| model_299 (final) | 0.123 m/s | 4.6 deg | 0 |

Training at iterations 250-299 (measured): episodes 999/1000, tracking error 0.025 m/s, torque saturation 3%, std 0.06.
Ablation result: bc1 (the first success) also carried a "linear progress" reward and a velocity low-pass filter left
over from the failed experiments; bcmin runs without them and tracks direction far better (5 deg vs 23 deg at
iteration 200). The progress term paid for any motion component along the command and ignored sideways drift.
They are therefore not part of the method.

## 4. Two gotchas that cost time
- **Stale optimizer state.** Splicing new actor weights under Adam moments computed for the old weights steers the
  first update wrongly; `bc_clone.py` clears `optimizer_state_dict["state"]`.
- **Starting several Isaac Sim processes in the same second** on the shared GPU crashed one of them once (carb
  `getenv` crash during startup). Stagger launches by a few seconds.

## 5. Known limits
Forward/backward commands only (0.10-0.15 m/s) and small yaw; ~21 degrees of sideways drift remains; the gait came
from the club's scripted tripod. Next: extend the teacher to lateral/yaw commands and re-clone.
