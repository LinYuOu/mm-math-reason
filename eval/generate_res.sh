# ps aux --sort=-%mem | awk 'BEGIN { FS = "[ \t]+" } /python/ && $5>=700000 { system("kill -9 " $2) }'
# 开启 errexit 模式
set -e

ray stop
export DISABLE_TORCHVISION_IMPORTS=1

export SYSTEM_PROMPT_RL="You are a multimodal reasoner. The user asks a question, and you solve it. You need first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
export SYSTEM_PROMPT_QwenVL="Let's think step by step.The final answer MUST BE put in \\boxed{}."
export SYSTEM_PROMPT_SFT_RL="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."

# step1: generate response
export ROOT_PATH='OpenCompassEval/Benchmarks'
export SYSTEM_PROMPT="${SYSTEM_PROMPT_SFT_RL}"
export MODEL_NAME_AND_PATH="../models/Qwen2.5-VL-3B-Instruct/"
export MAX_OUTPUT_TOKEN=128
OUTPUT_DIR="output/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR/log
log_path="${OUTPUT_DIR}/output.log"
echo "log name: $log_path"
# 复制当前脚本到日志目录
SCRIPT_NAME=$(basename "$0")
cp "$0" "${OUTPUT_DIR}/${SCRIPT_NAME}"

touch $log_path
python eval/generate_response.py \
    --model_name_and_path $MODEL_NAME_AND_PATH \
    --output_dir  $OUTPUT_DIR\
    --first_n 200 \
    2>&1 | tee -a ${log_path} \

# # step2: eval with math verify
# python eval/eval_from_jsonl_with_mathverify.py \
#     --input_file  \
#     --output_folder  \
