#!/bin/bash
# +
VERSION="bs_17k_fully_truncated"

export http_proxy=http://oversea-squid4.sgp.txyun:11080 https_proxy=http://oversea-squid4.sgp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online

accelerate launch \
    --config_file /data/kousiqi/nips/LLaMA-Factory/examples/accelerate/fsdp_config_8.yaml \
    --main_process_port 29501 \
    src/train.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --upcast_layernorm true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset bs_17k_fully_truncated \
    --template qwen25 \
    --cutoff_len 16384 \
    --max_samples 1000000 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 1 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --tf32 True \
    --val_size 0 \
    --output_dir /data/gexiang/kousiqi/checkpoints/qw25/$VERSION \
