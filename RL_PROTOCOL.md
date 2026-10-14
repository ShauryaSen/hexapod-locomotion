# A protocol for training locomotion with RL (and knowing when RL alone won't work)

Lesson from the hexapod: we spent 12 training runs on reward tweaks after the data had already shown the problem
wasn't the reward. This is the order of questions that would have cut that to ~4 runs. Every step has a cheap test
you run **before** the next expensive training run, and a rule for what the answer decides.

## The five questions

| # | Question | Test (minutes, not hours) | Hexapod answer (measured) | What it decides |
|---|---|---|---|---|
| 1 | **Feasibility.** Can the robot do the task at all in this sim? | Drive it with a scripted / classical controller (an "oracle"). | Scripted tripod walks 0.09–0.13 m/s, 0 falls. | **No** → fix the robot model or the sim, not the learning. |
| 2 | **Reward audit.** Does your reward prefer the behavior you want over the trivial behavior? | Score the oracle *and* the trivial policy (standing still) with the exact reward function. | Walking 1.17/s vs standing 1.13/s; most gaits scored *below* standing. | **No** → fix the reward before training anything. |
| 3 | **Signal under exploration.** Does the noisy policy ever produce the behavior, and does the gradient point at it? | Roll out the policy with and without exploration noise; compare. | The "ratchet": noisy policy drifts 0.06 m/s, noise-free policy stands still, no action direction is preferred. | **No** → stop tuning rewards. Add structure (step 5). |
| 4 | **Budget bound.** How long should a from-scratch run take if it's going to work? | Find a known-working recipe on a comparable robot and copy its step budget. | ANYmal flat walks within ~10–50M env steps; our runs got 2–4M each. | Sets the **kill rule** in steps, written down before launch. |
| 5 | **Method choice.** Which structure, given what failed? | The ladder, least prior first: reward shaping → curriculum → demos as *data* (BC/DAgger) → demos as *reward* (DeepMimic/AMP) → prior in the loop (residual/CPG). | Shaping and curricula failed; BC+PPO worked; DeepMimic too slow; residual worked but leaves a script in the controller. | Pick the **least** prior that passes step 3. |

## Rules that make it orderly

1. **Judge only the noise-free policy, from a standing start, with a direction check.** Training curves and
   "mean speed" lied to us repeatedly: exploration noise earns reward that the deployable policy never does.
   Use `logs/overnight/evalckpt.sh <run> <checkpoint>`: it prints speed *along the command*, heading error, falls.
2. **Write the kill rule before launching.** "Dead if the noise-free rollout shows < X m/s by N steps." Then obey it.
3. **One change per run.** Otherwise you can't tell what helped.
4. **A reward change is allowed only while step 3 still passes.** Once exploration can't produce the behavior, more
   reward engineering is head-banging; the next move is structure.
5. **Keep a table.** Run, one-line change, expected signal, measured result, verdict, video path.
   Ours is `logs/overnight/README.md`.

## Why step 3 fails on a hexapod specifically

PPO explores with independent noise on each joint. A tripod gait needs ~12 joints moving with the right signs, in
phase, for ~20 consecutive steps; independent noise essentially never does that. On a quadruped, balance *forces*
stepping, and the reward then shapes those steps. A hexapod on five legs is perfectly stable, so standing is never
punished and nothing forces coordination. The noisy policy finds a posture where random twitches net forward drift,
every twitch pays, and the mean action has no reason to move. That is the measured mechanism, not a guess.

## What worked, in one paragraph

Give the network the gait first, then let PPO improve it. Clone the scripted tripod into the PPO actor by
supervised learning (behavior cloning), fix the compounding errors with two DAgger rounds (let the clone drive,
relabel the states it reaches with the teacher, refit), splice the weights into an rsl_rl checkpoint, and fine-tune
with PPO at low learning rate and small exploration std. Result: 0.123 m/s along the command, 4.6 degrees heading
error, 0 falls, from a plain MLP with nothing scripted at runtime. Full recipe, every code change, and the reproduce
commands: [`METHOD.md`](METHOD.md). Code: [`tools/bc_clone.py`](tools/bc_clone.py).

## Reading list (from memory; check the links)

- Schulman, *The Nuts and Bolts of Deep RL Research* (2016 talk): oracle tests, reward scales, look at rollouts.
- OpenAI *Spinning Up*, "Spinning Up as a Deep RL Researcher": research discipline.
- Ibarz et al., *How to train your robot with deep RL: lessons we've learned* (2021): when demos beat shaping.
- Ng, Harada, Russell, *Policy invariance under reward transformations* (1999): the only shaping that can't be gamed.
- Ross, Gordon, Bagnell, *DAgger* (2011) · Peng et al., *DeepMimic* (2018), *AMP* (2021) · Uchendu et al., *Jump-Start RL* (2023).
- Rudin et al., *Learning to Walk in Minutes* (2022): the from-scratch quadruped recipe and why it works there.
