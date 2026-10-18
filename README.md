# hexapod

A minimal setup for training a walking policy for a six-legged robot in Isaac Sim / Isaac Lab: a direct-RL
velocity-tracking environment (adapted from Isaac Lab's ANYmal-C locomotion task) with PD-controlled joints,
contact sensing, body-mass domain randomization and PPO (rsl_rl).

![hexapod walking](media/walking.gif)

**Results (simulation):** PPO from scratch learns to stand but not to walk. A six-legged robot is stable standing
still, so exploration never finds a coordinated stride. Cloning a scripted tripod gait into the policy network (BC +
DAgger) and fine-tuning it with PPO gives a plain MLP that walks on command at 0.12 m/s with 5° heading error and
no falls. A simplified collision model (19 primitives instead of 153 meshes) speeds up simulation 2.4–4.5×.

The robot was designed by the Cornell Physical Intelligence (CUPI) hexapod team. We extend this setup and are
currently researching different locomotion methods in the CUPI repo:
<https://github.com/Cornell-Physical-Intelligence/hexapod-cupi>.

## Run

Requires Isaac Lab 3.0. Clone this repo as a folder named `hexapod` and put its parent directory on `PYTHONPATH`;
then, from the Isaac Lab root:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Hexapod-Flat-Direct-v0 \
    --external_callback hexapod.register --headless --num_envs 512
```
