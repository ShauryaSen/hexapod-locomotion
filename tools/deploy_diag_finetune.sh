#!/usr/bin/env bash
# "Is data the bottleneck?" diagnostic (Shaurya: diagnose first, 2026-10-23). Fine-tune pi_0 at a fixed motor
# strength with the full training setup (train.py, 512 envs, the run's own terrain / reward / DR / PPO settings from
# params/, plain PPO, no KL anchor), ~300 iterations, checkpoints every 50; then evaluate every checkpoint and the
# frozen pi_0 at that motor strength on the deployment terrain (levels 0-5, all types, cmd 0.28) with jittered
# starts. Prediction and decision rule: harness-report.md, "Diagnostic: data or physics".
#
# usage (on the Spark, from ~/src/isaaclab-learning):  hexapod/tools/deploy_diag_finetune.sh [--dry_run]
# env: DIAG_MOTOR (0.6), DIAG_ITERS (300), DIAG_SAVE (checkpoint interval, 50), DIAG_RPC (eval robots per cell, 8),
#      DIAG_NAME (run name / eval tag). Sample-efficiency curve: DIAG_ITERS=50 DIAG_SAVE=5 DIAG_RPC=4 DIAG_NAME=diagcurve_motor0.6_7299
set -u
cd ~/src/isaaclab-learning
RUN=2026-10-22_06-52-05_stage1v21_seed1
CK=model_7299.pt
MOTOR=${DIAG_MOTOR:-0.6}
ITERS=${DIAG_ITERS:-300}
SAVE=${DIAG_SAVE:-50}
RPC=${DIAG_RPC:-8}
NAME=${DIAG_NAME:-diag_motor${MOTOR}_7299}
B=/workspace/isaaclab/logs/rsl_rl/hexapod_rough_direct
OV=$(~/bin/isaac-shell /workspace/isaaclab/_isaac_sim/python.sh /workspace/learning/hexapod/tools/deploy_diag_overrides.py --run_dir $B/$RUN --dr motor_strength_range=$MOTOR,$MOTOR 2>/dev/null | tail -1)
echo "[diag] overrides: $OV"
if [ "${1:-}" = "--dry_run" ]; then exit 0; fi
export SPARK_RUN_STALL_S=0
# 1. fine-tune (resume from pi_0; rsl_rl continues the iteration count from 7299)
hexapod/tools/spark_run.sh logs/deploy/${NAME}_train.log -- env PYTHONPATH=/workspace/learning ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Hexapod-Rough-Direct-v0 --external_callback hexapod.register --headless --num_envs 512 --seed 1 \
  --max_iterations $ITERS --resume --load_run $RUN --checkpoint $CK --run_name $NAME $OV agent.save_interval=$SAVE
NEW=$(ls -td $HOME/src/isaaclab-learning/logs/rsl_rl/hexapod_rough_direct/*_${NAME} | head -1)
echo "[diag] fine-tune run: $NEW"
sleep 60
# 2. evaluate the frozen pi_0 and every saved fine-tune checkpoint at the same motor strength (params from the
#    original run, so terrain / commands are identical; jittered starts so the robots of a cell are not copies)
EVAL="--task Hexapod-Rough-Direct-v0 --dr_cells 1.0,0.0,$MOTOR --levels 0 1 2 3 4 5 --cmds 0.28 --robots_per_cell $RPC --num_envs 512 --start_jitter 0.05 10 --seed 0"
hexapod/tools/spark_run.sh logs/deploy/${NAME}_eval_frozen.log -- env PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/eval_suite.py --headless \
  --checkpoint $B/$RUN/$CK --run_dir $B/$RUN $EVAL --tag ${NAME}
sleep 60
hexapod/tools/spark_run.sh logs/deploy/${NAME}_eval_ft.log -- env PYTHONPATH=/workspace/learning ./isaaclab.sh -p /workspace/learning/hexapod/tools/eval_suite.py --headless \
  --run_dir $B/$(basename $NEW) $EVAL --tag ${NAME}
echo "[diag] done"
