#! /bin/bash
# DEVICE=cuda:0

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/play.py \
--task g1_distill \
--load_run Jun03_20-36-10_G1_9skills_fromMay28_21-08-07 \
--checkpoint 108000 \
# --headless
# --load_run Mar20_16-45-53_track_1.7 \