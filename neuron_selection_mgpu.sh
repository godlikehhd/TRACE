#!/bin/bash

gpu_list=(0)
data_size=67676

for gpu in "${gpu_list[@]}"; do
  start_index=$(((gpu) * data_size))
  end_index=$(((gpu + 1) * data_size))
  index=$((gpu))
  echo 'Running process #' ${index} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  
  (
    CUDA_VISIBLE_DEVICES=$gpu python neural_data_selection.py \
      --data_path 'datasets/data_repo/10k.json' \
      --save_path results/qwen-2.5-7B/batch_test/${index}_random  \
      --model_name_or_path_source Qwen2.5-7B-Instruct \
      --max_length 4096 \
      --start_idx $start_index \
      --end_idx $end_index
  ) &

done



# 等待所有后台任务结束
wait
