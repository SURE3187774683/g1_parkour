#! /bin/bash
# DEVICE=cuda:0

python /home/pc/workspace/g1_parkour/legged_gym/legged_gym/scripts/play.py \
--task g1_field \
--load_run May16_22-26-12_v1 \
--checkpoint 15500 \
# --headless

# --load_run Mar20_16-45-53_track_1.7 \