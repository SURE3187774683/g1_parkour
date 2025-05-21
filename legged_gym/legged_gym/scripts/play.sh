#! /bin/bash
# DEVICE=cuda:0

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/play.py \
--task g1_field \
--load_run May19_20-39-14_dof_error_named-0.2 \
--checkpoint 27500 \
# --headless

# --load_run Mar20_16-45-53_track_1.7 \