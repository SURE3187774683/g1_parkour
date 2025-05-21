#! /bin/bash

DEVICE=cuda:7

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/train.py \
--task g1_field \
--rl_device $DEVICE \
--sim_device $DEVICE \
--run_name dof_error_named-0.2_Nocontact_no_vel_min_feet_distance0.2_Nodof_pos_limits_Noalive_Nohip_pos \
--headless \
# --run_name From_Apr18_15-08-13_TerrainPerlin_50000 \
# --load_run=Apr15_14-51-52_Tron1_10skills_fromApr11_17-14-24 \
# --checkpoint=74300 \