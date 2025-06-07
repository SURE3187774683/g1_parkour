#! /bin/bash
# DEVICE=cuda:0

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/play.py \
--task g1_distill \
--load_run Jun05_19-10-22_G1_9skills_fromDistill_best \
--checkpoint 35000 \
--headless
# --load_run Mar20_16-45-53_track_1.7 \