MESA_VK_DEVICE_SELECT='10de:2684'
CUDA_VISIBLE_DEVICES=0

python /home/shaos/workspace/g1_parkour/legged_gym/legged_gym/scripts/collect.py \
--task g1_distill \
--log \
--load_run /home/shaos/workspace/g1_parkour/legged_gym/logs/distill_G1/Jun07_13-19-41_G1_9skills_fromDistill_best \
--headless \