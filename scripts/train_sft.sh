# A shell script to run SFT


export LOG_LEVEL='INFO'
ps aux --sort=-%mem | awk 'BEGIN { FS = "[ \t]+" } /python/ && $5>=700000 { system("kill -9 " $2) }'
ray stop
cd mm_math_reason/scripts/

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export EXPERIMENT_NAME=debug
export BASE_MODEL_PATH="models/Qwen2.5-3B-Instruct"
export DATA="SFT.jsonl"
export OUTPUT_DIR="output/$EXPERIMENT_NAME"


TIMESTAMP=$(date +%Y%m%d_%H%M)
# 创建日志目录结构
LOG_DIR="log/${EXPERIMENT_NAME}/"
mkdir -p ${LOG_DIR}

# 生成带时间戳的日志文件名
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

# 复制当前脚本到日志目录
SCRIPT_NAME=$(basename "$0")
cp "$0" "${LOG_DIR}/${SCRIPT_NAME}"

MAX_PIXELS=1003520 \
swift sft \
    --model $BASE_MODEL_PATH \
    --dataset $DATA \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 32 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir  $OUTPUT_DIR\
    --system mm_math_reason/scripts/prompt.txt \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero1 \
    --save_only_model true \
    --report_to tensorboard \
    --attn_impl flash_attn \
    --train_type full \
    2>&1 | tee -a ${LOG_FILE}
    # --train_type lora \
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # # --target_modules all-linear \
