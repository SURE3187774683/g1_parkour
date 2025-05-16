#! /bin/bash
# DEVICE=cuda:0

python /home/pc/workspace/h1_parkour/legged_gym/legged_gym/scripts/play.py \
--task H1_field \
--load_run Apr24_09-05-35_v2 \
--checkpoint 350000 \
# --headless

# --load_run Mar20_16-45-53_track_1.7 \