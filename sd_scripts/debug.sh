#!/bin/bash

# ==================== Config ====================
init_config=/mnt/hdfs/seed_wl/jianxiaowen.24/InternVL2-26B/
model_path=/mnt/hdfs/seed_wl/jianxiaowen.24/tuwen_vlm/intern_dual_vlm189.bin
base_output=/mnt/hdfs/user/huangrunhui/exps/reward_models
plot_dir=./plots

# Data paths
meigan_json=/mnt/hdfs/seed_wl/gaoyu.123/data/meigan_data1/data.json
tuwen_json=/mnt/hdfs/seed_wl/gaoyu.123/data/tuwen_data1/data.json
jiegou_json=/mnt/hdfs/seed_wl/gaoyu.123/data/jiegou_data1/data.json

# Output dirs
meigan_output=${base_output}/meigan_eval
tuwen_output=${base_output}/tuwen_eval
jiegou_output=${base_output}/jiegou_eval

# ==================== Inference ====================
torchrun --master_port=25641 --nproc-per-node=4 internvl2_infer_t2i.py \
  --json_input ${meigan_json} \
  --init_config ${init_config} \
  --model_path ${model_path} \
  --output_dir ${meigan_output}

torchrun --master_port=25642 --nproc-per-node=4 internvl2_infer_t2i.py \
  --json_input ${tuwen_json} \
  --init_config ${init_config} \
  --model_path ${model_path} \
  --output_dir ${tuwen_output}

torchrun --master_port=25643 --nproc-per-node=4 internvl2_infer_t2i.py \
  --json_input ${jiegou_json} \
  --init_config ${init_config} \
  --model_path ${model_path} \
  --output_dir ${jiegou_output}

# ==================== Visualization ====================
# Single dataset
python plot_reward_scores.py \
  --input_files ${meigan_output} \
  --labels "美感" \
  --output ${plot_dir}/meigan.png

python plot_reward_scores.py \
  --input_files ${tuwen_output} \
  --labels "图文" \
  --output ${plot_dir}/tuwen.png

python plot_reward_scores.py \
  --input_files ${jiegou_output} \
  --labels "结构" \
  --output ${plot_dir}/jiegou.png

# Combined
python plot_reward_scores.py \
  --input_files ${meigan_output} ${tuwen_output} ${jiegou_output} \
  --labels "美感" "图文" "结构" \
  --output ${plot_dir}/combined.png
