pip install -r requirements.txt

cd mm_math_reason/ms-swift
pip install -e .
pip install -U deepspeed trl qwen-vl-utils vllm==0.7.3 ms-swift==3.3.0.post1
# pip install "deepspeed<0.17"