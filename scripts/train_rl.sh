export LOG_LEVEL='INFO'
ps aux --sort=-%mem | awk 'BEGIN { FS = "[ \t]+" } /python/ && $5>=700000 { system("kill -9 " $2) }'
ray stop
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=7


export DATA="items.jsonl"


export MODEL_PATH="models/Qwen2.5-VL-7B-Instruct"

export EXPERIMENT_NAME=

cd mm_math_reasoning/scripts/ # root path
# 创建日志目录结构
LOG_DIR="log/${EXPERIMENT_NAME}/"
mkdir -p ${LOG_DIR}
# 生成日志文件名
LOG_FILE="${LOG_DIR}/train.log"
# 复制当前脚本到日志目录
SCRIPT_NAME=$(basename "$0")
cp "$0" "${LOG_DIR}/${SCRIPT_NAME}"
OUTPUT_DIR=output/$EXPERIMENT_NAME

MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type grpo \
    --model  $MODEL_PATH\
    --external_plugins mm_math_reason/scripts/plugin.py \
    --reward_funcs external_answer_tag_acc\
    --use_vllm true \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --vllm_limit_mm_per_prompt '{"image": 10}' \
    --train_type full \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --dataset  $DATA2\
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --eval_strategy 'no' \
    --num_generations 4 \
    --per_device_train_batch_size  4\
    --gradient_accumulation_steps  2\
    --learning_rate 1e-6 \
    --max_grad_norm 1.0 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 4 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --async_generate false \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 4 \
    --num_infer_workers 1 \
    --report_to tensorboard \
    --output_dir  $OUTPUT_DIR\
    --save_only_model true \
    --overlong_filter true \
    --beta 2e-3 \
    --system mm_math_reason/scripts/prompt_sft.txt \
    --epsilon_high	0.28 \
    2>&1 | tee -a ${LOG_FILE}
    # --dynamic_sample true \
    # --max_resample_times 3\
    # --epsilon_high 0.28\
    # --loss_type bnpo\
    # --soft_cache_length 1638\
