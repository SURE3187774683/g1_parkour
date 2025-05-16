#! /bin/bash

DEVICE=cuda:0

python /home/pc/workspace/h1_parkour/legged_gym/legged_gym/scripts/train.py \
--task H1_field \
--rl_device $DEVICE \
--sim_device $DEVICE \
--run_name plane_-base_height_-feet_swing_height_-lin_vel_z_reward_tracking_lin_vel \
# --headless \
# --run_name From_Apr18_15-08-13_TerrainPerlin_50000 \
# --load_run=Apr15_14-51-52_Tron1_10skills_fromApr11_17-14-24 \
# --checkpoint=74300 \