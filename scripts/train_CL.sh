# a shell sciript to run a multi-stage training process for curriculum learning

export LOG_LEVEL='INFO'
ps aux --sort=-%mem | awk 'BEGIN { FS = "[ \t]+" } /python/ && $5>=700000 { system("kill -9 " $2) }'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=7

export STAGE1_DATA="Dataset/CL/geo_text_8k/20250604_151747/merged_simple_questions.jsonl"
export STAGE2_DATA="Dataset/CL/geo_text_8k/20250604_151747/merged_medium_questions.jsonl"
export STAGE3_DATA="Dataset/CL/geo_text_8k/20250604_151747/merged_difficult_questions.jsonl"

# stage1
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model /group/40174/druryxu/mm_reason/checkpoints/geo-text-problem/v4-20250518-233141/checkpoint-900 \
    --external_plugins /group/40078/druryxu/math_to_mllm/scripts/plugin.py \
    --reward_funcs external_answer_tag_acc format\
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_limit_mm_per_prompt '{"image": 10}' \
    --train_type full \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --dataset  $STAGE1_DATA\
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --eval_strategy 'no' \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --max_grad_norm 0.2 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --async_generate false \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --report_to tensorboard \
    --output_dir output/stage1 \
    --save_only_model true \
    --overlong_filter true \
    --beta 0 \
    --system scripts/prompt_rl.txt \

# stage2
echo "等待2分钟，确保stage1模型保存完成..."
sleep 2m
# 获取stage1下最新的子文件夹
latest_subfolder=$(ls -t output/stage1 | head -n 1)
echo "最新子文件夹: $latest_subfolder"

# 获取该子文件夹中编号最大的checkpoint
checkpoint_folder=$(ls -d output/stage1/"$latest_subfolder"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo "checkpoint_folder: $checkpoint_folder"

# 检查是否找到checkpoint文件夹
if [ -z "$checkpoint_folder" ]; then
    echo "错误: 在 output/stage1/$latest_subfolder 中未找到checkpoint文件夹"
    exit 1
fi

# 设置MODEL_PATH为找到的checkpoint路径
export MODEL_PATH2="$checkpoint_folder"
export MODEL_PATH2=output/stage1/v17-20250527-165239/checkpoint-170

# 后续使用$MODEL_PATH的命令保持不变
echo "使用的模型路径: $MODEL_PATH2"
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model  $MODEL_PATH2\
    --external_plugins /group/40078/druryxu/math_to_mllm/scripts/plugin.py \
    --reward_funcs external_answer_tag_acc format\
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_limit_mm_per_prompt '{"image": 10}' \
    --train_type full \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --dataset $STAGE2_DATA \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --eval_strategy 'no' \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --max_grad_norm 0.2 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --async_generate false \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --report_to tensorboard \
    --system scripts/prompt_rl.txt \
    --output_dir output/stage2 \
    --save_only_model true \
    --overlong_filter true \
    --beta 0

# stage3
echo "等待2分钟，确保stage2模型保存完成..."
sleep 2m
# 获取stage2下最新的子文件夹
latest_subfolder=$(ls -t output/stage2 | head -n 1)
echo "最新子文件夹: $latest_subfolder"

# 获取该子文件夹中编号最大的checkpoint
checkpoint_folder=$(ls -d output/stage2/"$latest_subfolder"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo "checkpoint_folder: $checkpoint_folder"

# 检查是否找到checkpoint文件夹
if [ -z "$checkpoint_folder" ]; then
    echo "错误: 在 output/stage2/$latest_subfolder 中未找到checkpoint文件夹"
    exit 1
fi

# 设置MODEL_PATH为找到的checkpoint路径
export MODEL_PATH3="$checkpoint_folder"

# 后续使用$MODEL_PATH的命令保持不变
echo "使用的模型路径: $MODEL_PATH3"
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model  $MODEL_PATH3\
    --external_plugins /group/40078/druryxu/math_to_mllm/scripts/plugin.py \
    --reward_funcs external_answer_tag_acc format\
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 4096 \
    --vllm_limit_mm_per_prompt '{"image": 10}' \
    --train_type full \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --dataset $STAGE3_DATA \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --eval_strategy 'no' \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --max_grad_norm 0.2 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --async_generate false \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --report_to tensorboard \
    --system scripts/prompt_rl.txt \
    --output_dir output/stage3 \
    --save_only_model true \
    --overlong_filter true \
    --beta 0

# # # eval
echo "等待2分钟，确保stage3 模型保存完成..."
sleep 2m

# python run_gpu.py

# 获取stage3下最新的子文件夹
latest_subfolder=$(ls -t output/stage3 | head -n 1)
echo "最新子文件夹: $latest_subfolder"

# 获取该子文件夹中编号最大的checkpoint
checkpoint_folder=$(ls -d output/stage3/"$latest_subfolder"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
echo "checkpoint_folder: $checkpoint_folder"

# 检查是否找到checkpoint文件夹
if [ -z "$checkpoint_folder" ]; then
    echo "错误: 在 output/stage3/$latest_subfolder 中未找到checkpoint文件夹"
    exit 1
fi

# 设置MODEL_PATH为找到的checkpoint路径
export MODEL_PATH3="$checkpoint_folder"
python eval/eval.py \
    --model_name_and_path $MODEL_PATH3 \
    --device 0 \
    --output_dir output/stage3
