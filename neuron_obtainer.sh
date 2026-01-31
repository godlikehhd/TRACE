CUDA_VISIBLE_DEVICES=0 python neuron_obtainer.py \
    --data_path './datasets/demo/tydiqa_val.json' \
    --save_path ./neuron_activations/llama-3.1-8B/tydiqa/base \
    --model_name_or_path cluster_0_10000/checkpoint-79 \
    --max_length  4096 


CUDA_VISIBLE_DEVICES=1 python neuron_obtainer.py \
    --data_path 'datasets/demo/tydiqa_val.json' \
    --save_path neuron_activations/llama-3.1-8B/tydiqa/val \
    --model_name_or_path tydiqa_val_cluster_ckpt1/checkpoint-1 \
    --max_length  4096 

