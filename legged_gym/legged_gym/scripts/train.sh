#! /bin/bash

DEVICE=cuda:2

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/train.py \
--task g1_field \
--rl_device $DEVICE \
--sim_device $DEVICE \
--run_name v1_based_May23_22-18-37_parlin_v1_feet_air_time10_ang_vel_xy-0.4 \
--headless \
# --run_name From_Apr18_15-08-13_TerrainPerlin_50000 \
# --load_run=Apr15_14-51-52_Tron1_10skills_fromApr11_17-14-24 \
# --checkpoint=74300 \