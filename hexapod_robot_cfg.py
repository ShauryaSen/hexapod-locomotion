# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the club hexapod (MkII, RS05 mass-corrected model).

Values are ported from the club repo (~/src/hexapod):

* USD: robot/hexapod_mkii_updated_v1/usd/ (copied into ./assets/usd/)
* Joint/link names: locomotion/env_config.py
* Standing pose and reset height: robot/hexapod_mkii_updated_v1/stance.json
* Motor gains and torque cap: locomotion/env.py (motor_force) and env_config.py (KD)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

HEXAPOD_USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "usd", "robot.usda")

HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HEXAPOD_USD_PATH,
        activate_contact_sensors=True,
        # The club USD deliberately has no DriveAPI on its joints (their code applies torques by hand).
        # Without one, PhysX ignores the actuator's PD gains below; this adds a force drive to each joint.
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.103),
        joint_pos={
            ".*_coxa_yaw": 0.0,
            ".*_femur_pitch": -0.3,
            ".*_tibia_pitch": 0.4,
        },
    ),
    actuators={
        # PD servo: kp = 12 N·m/rad for every joint, per-joint kd, 1.6 N·m cap.
        # The club model also lowers the cap above ~450 rpm; that speed curve is not modeled here.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=12.0,
            damping={
                ".*_coxa_yaw": 0.442,
                ".*_femur_pitch": 0.246,
                ".*_tibia_pitch": 0.106,
            },
            effort_limit_sim=1.6,
            velocity_limit_sim=50.27,
        ),
    },
)
"""Configuration of the club hexapod with an implicit PD actuator."""
