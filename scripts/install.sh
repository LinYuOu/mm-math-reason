export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcos.cn,.myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

cd mm_math_reasoning/ms-swift
pip install -e .
pip install -U deepspeed trl qwen-vl-utils vllm==0.7.3 ms-swift==3.3.0.post1
# pip install "deepspeed<0.17"