#!/bin/bash
# +
VERSION="gemini_flash_2047"

export http_proxy=http://oversea-squid4.sgp.txyun:11080 https_proxy=http://oversea-squid4.sgp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-12.4/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --main_process_port 29501 \
    src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --model_name_or_path /data/phd/jinjiachun/ckpt/Qwen/Qwen2.5-VL-7B-Instruct \
    --upcast_layernorm true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --dataset sft_gemini_flash_2047 \
    --template qwen2_vl \
    --cutoff_len 16384 \
    --max_samples 1000000 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --logging_steps 10 \
    --save_steps 40 \
    --save_total_limit 1 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --tf32 True \
    --val_size 64 \
    --per_device_eval_batch_size 8 \
    --eval_strategy steps \
    --eval_steps 40 \
    --output_dir /data/phd/jinjiachun/experiment/sft_qwenvl/$VERSION \
