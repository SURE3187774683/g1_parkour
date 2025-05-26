#! /bin/bash
# DEVICE=cuda:0

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/play.py \
--task g1 \
--load_run May24_11-23-46_v1_energy_dof_error_names+hip_roll \
--checkpoint 1000 \
# --headless

# --load_run Mar20_16-45-53_track_1.7 \