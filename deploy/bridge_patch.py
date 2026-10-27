# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""train.py ``--external_callback`` for the bridge run (deployment updater settings applied one at a time to the
512-env fine-tune, 2026-10-27): registers the hexapod tasks and, if ``BRIDGE_PIN_CMD`` is set (m/s), pins every
robot's command after each reset to the deployment command (forward along -y at that speed, no yaw), exactly as
``deploy.envutil.pin_commands`` does in the harness. The command ranges stay as trained, so the reward's tracking
tolerances are unchanged. The env file itself is not edited (it belongs to the pi_0 session).

The callback runs before the simulation app starts, so the env class is patched lazily: the tasks' gym entry point
is replaced by a function that patches the class on first construction.

  BRIDGE_PIN_CMD=0.28 train.py ... --external_callback hexapod.deploy.bridge_patch.register
"""

import os

import gymnasium as gym


def _make(**kwargs):
    from hexapod.deploy.envutil import pin_commands
    from hexapod.hexapod_env import HexapodEnv

    speed = float(os.environ["BRIDGE_PIN_CMD"])
    if not getattr(HexapodEnv, "_bridge_pinned", False):
        orig = HexapodEnv._reset_idx

        def _reset_idx(self, env_ids):
            orig(self, env_ids)
            pin_commands(self, speed, env_ids)

        HexapodEnv._reset_idx = _reset_idx
        HexapodEnv._bridge_pinned = True
        print(f"[bridge] commands pinned to {speed} m/s forward, 0 yaw after every reset")
    return HexapodEnv(**kwargs)


def register():
    import hexapod  # noqa: F401  (gym.register of the tasks)

    if os.environ.get("BRIDGE_PIN_CMD"):
        for task_id, spec in gym.registry.items():
            if task_id.startswith("Hexapod-") and str(spec.entry_point).endswith("hexapod_env:HexapodEnv"):
                spec.entry_point = _make
        print(f"[bridge] pinning enabled for the hexapod tasks ({os.environ['BRIDGE_PIN_CMD']} m/s)")
