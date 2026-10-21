# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase B deployment harness (cs6758 decisions.md F2, F5-F7, E1-E4).

N independent single-robot deployments (k = 1) run in one Isaac Lab process, each with its own policy copy and
optimizer state. The modules here are plain torch and importable without Isaac Sim; only ``deploy.runner`` and
``deploy.envutil`` touch the env, and they import it lazily (after ``AppLauncher``).

  cfg.py          every tunable, one labelled field each (MEASURED / CITED / ASSUMED in the docstring)
  batched.py      stacked-parameter MLPs (one weight slice per deployment), batched Adam, checkpoint loading
  updater.py      KL-anchored PPO on actor + critic, one deployment's own stream only (F7)
  ensemble.py     5-member dynamics ensemble and the per-gait-cycle residual (F5)
  detectors.py    3-sigma rule, CUSUM, convergence stop
  arms.py         adaptation-timing arms as wrappers around the same updater (F6)
  schedule.py     Poisson change schedules (E1) and their application through the env's runtime setters
  streams.py      per-deployment gait-cycle streams (progress, tracking error, reward, falls, residual, state)
  metrics.py      E2 metrics from streams + schedules; bootstrap CIs
  runner.py       the deployment loop
"""
