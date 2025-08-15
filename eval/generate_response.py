"a file to generate mllm responses for evaluation"
import math
import re
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# from vllm.lora.request import LoRARequest
from qwen_vl_utils import process_vision_info
import ray

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate
from datasets import load_dataset

from math_verify import (
    StringExtractionConfig,
    ExprExtractionConfig,
    LatexExtractionConfig,
)
from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig


SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "let's think step by step. The final answer MUST BE put in \\boxed{}.",
)
print(f"SYSTEM_PROMPT: {SYSTEM_PROMPT}")

ROOT = os.environ.get("ROOT_PATH", "eval/Benchmarks")
MAX_OUTPUT_TOKEN = int(os.environ.get("MAX_OUTPUT_TOKEN", 1024))  # 设置最大输出长度


def create_parser():
    """Creates a command-line argument parser."""

    parser = argparse.ArgumentParser(description="A simple example parser.")

    # Add arguments
    parser.add_argument("--model_name_and_path", help="Path to the model.")
    parser.add_argument("--lora_path", help="Path to the lora model.")
    parser.add_argument("--device", help="Device to use for inference.")
    parser.add_argument("--output_dir", help="Path to result.")
    parser.add_argument("--first_n", help="是否取 前 n 个数据")
    parser.add_argument(
        "--save_batch_size",
        type=int,
        default=1024,
        help="Batch size for saving results.",
    )

    return parser


def process_example(example, name, processor):
    if name == "OlympiadBench":
        if "_TO_" in example["from"]:
            if example["context"] is not None:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": example["context"] + "\n\n" + example["question"],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ]
        else:
            items = re.split("(<img_\d+>)", example["question"])
            items = [item for item in items if item and not item.isspace()]
            image_root = "OlympiadBench/OlympiadBench_Dataset/images"

            messages = [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.join(image_root, item[1:-1] + ".jpg"),
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        }
                        for item in items
                        if item[:5] == "<img_"
                    ]
                    + [{"type": "text", "text": example["question"]}],
                },
            ]
    elif name == "MATH-V":
        question = example["question"]

        # pattern = r'(\n<image\d*>)+$'
        # question = re.sub(pattern, '', question)

        image_root = os.path.join(ROOT, "MATH-V")
        image = os.path.join(image_root, example["image"])

        if len(example["options"]) > 0 and "".join(example["options"]) != "ABCDE":
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"

            question += f"\n{options}"

    elif name == "We-Math":
        question, options = example["question"], example["option"]

        image_root = os.path.join(ROOT, "We-Math/data")
        image = os.path.join(image_root, example["image_path"])

        question += f"\n{options}"

    elif name == "MathVerse":
        question = example["question_for_eval"]

        image_root = os.path.join(ROOT, "MathVerse/images")
        image = os.path.join(image_root, example["image"])

    elif name == "MathVista":
        question = example["question"]

        image_root = os.path.join(ROOT, "MathVista")
        image = os.path.join(image_root, example["image"])

        if example["choices"]:
            question += (
                "\n\nPlease select the correct answer from the following options:"
            )
            options = "\n".join(
                [
                    f"({chr(ord('A') + i)}) {option}"
                    for i, option in enumerate(example["choices"])
                ]
            )
            question += f"\n{options}"

    elif name == "LogicVista":
        question = example["question"]

        image_root = os.path.join(ROOT, "LogicVista/data/images")
        image = os.path.join(image_root, example["imagename"])

    elif name == "DynaMath":
        # print(example)
        question = example["question"]

        image_root = os.path.join(ROOT, "DynaMath/samples_and_result/10trials")
        image = os.path.join(image_root, example["image"])
        # image = example["decoded_image"].convert("RGB")
        # example.pop("decoded_image")

    example["question"] = question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 256 * 28 * 28,  # image最小占256左右个token
                    "max_pixels": 1280 * 28 * 28,  # image最大占1280左右个token
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 构造仅包含视觉标记和问题文本的prompt
    # prompt = f"<|vision_start|><|image_pad|><|vision_end|>{question}"

    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs

    # if example["id"] == "327":
    #     print("example:", example)
    #     print("prompt: ", prompt)
    #     print("327: ", mm_data)
    # if example["id"] == "1421":
    #     print("example:", example)
    #     print("prompt: ", prompt)
    #     print("1421: ", mm_data)
    # assert 0
    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }, example


def extract_options(text, name):
    # 定义正则表达式模式
    if name == "We-Math":
        pattern = r"([A-E])\.\s*(.*?)(?=;|$)"
    elif name == "MathVerse":
        pattern = r"([A-E]):\s*(.*?)(?=;|$)"

    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, text)

    # 创建结果字典
    results = {}
    for option, content in matches:
        results[option] = content.strip()

    return results


def extract_boxed_content(content):
    """提取文本中\boxed{}包裹的内容"""
    # 找到\boxed{的位置
    left = content.rfind("\\boxed{")
    if left == -1:
        return None  # 没找到\boxed{

    # 括号开始位置 (跳过\boxed{)
    start_pos = left + len("\\boxed{")

    # 用栈进行括号匹配，处理嵌套括号
    stack = 1  # 已经找到一个左括号
    right = start_pos

    while right < len(content) and stack > 0:
        if content[right] == "{":
            stack += 1
        elif content[right] == "}":
            stack -= 1
        right += 1

    # 检查是否成功找到匹配的右括号
    if stack != 0:
        return None  # 括号不匹配

    # 获取\boxed{}中的内容 (不包括花括号)
    boxed_content = content[start_pos : right - 1]

    return boxed_content


def fix_latex_answer(answer):
    """
    修复LaTeX表达式:
    1. 确保表达式被$符号包裹
    参数:
        answer (str): 输入的LaTeX表达式，例如 "\\frac{1}{2}" 或 "1.0"
    返回:
        str: 修复后的LaTeX表达式"$\\frac{1}{2}$" 或 "$1.0$"
    """
    # 如果答案为空，直接返回
    if not answer:
        return answer

    # 检查是否已经被$符号包裹
    if not (answer.startswith("$") and answer.endswith("$")):
        answer = f"${answer}$"
    return answer

def evaluate(all_preds, name):
    total, correct = 0, 0
    # answer_prefixes = ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']

    parsed_model_answers, correctness = [], []
    # print("idx", idx)
    print("len all_preds", len(all_preds))
    for line in tqdm(all_preds):
        if name in [
            "MATH-V",
            "We-Math",
            "MathVerse",
            "MathVista",
            "LogicVista",
            "DynaMath",
        ]:
            answer = line["answer"]
        # elif name in ["DynaMath"]:
        #     answer = line[""]

        model_answer = line["resp"]
        if "<answer>" in SYSTEM_PROMPT:
            # 尝试匹配 <answer> 标签
            model_answer_match = re.search(r"<answer>(.*?)</answer>", model_answer)
        else:
            model_answer_match = extract_boxed_content(model_answer)

        # 尝试匹配 <answer> 标签
        # model_answer_match = re.search(r'<answer>(.*?)</answer>', model_answer, re.DOTALL | re.MULTILINE)
        # if not model_answer_match:
        if model_answer_match:
            if isinstance(model_answer_match, re.Match):
                model_answer_match = model_answer_match.group(1).strip()
            else:
                model_answer_match = model_answer_match
        model_answer = (
            fix_latex_answer(model_answer_match)
            if model_answer_match
            else model_answer.strip()
        )

        parsed_model_answer = parse(
            model_answer,
            extraction_config=[
                StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                ExprExtractionConfig(),
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="last",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
            ],
            extraction_mode="first_match",
        )
        parsed_model_answers.append(str(model_answer))

        answer = fix_latex_answer(answer)
        parsed_answer = parse(
            answer,
            extraction_config=[
                StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ],
        )

        flag = verify(parsed_answer, parsed_model_answer) or answer == model_answer

        correct += int(flag)
        total += 1

        correctness.append(flag)

    if total == 0:
        return "0.0000", parsed_model_answers, correctness
    acc = format(100 * correct / total, ".4f")
    if name == "DynaMath":
        image_to_verify = {}

        for example, correct in zip(all_preds, correctness):
            # 提取image路径的后两段作为key
            img_path = example["image"]
            # 按路径分隔符拆分
            path_parts = img_path.split("/")  # 处理Unix风格路径
            # 取后两段组合成新key
            if len(path_parts) >= 2:
                key = "/".join(path_parts[-2:])  # 得到 "image/image499.png" 格式
            else:
                # 路径过短时直接使用原路径（避免索引错误）
                key = img_path

            if key not in image_to_verify:
                image_to_verify[key] = [correct]
            else:
                image_to_verify[key].append(correct)

        item_min_acc = []
        for k, v in image_to_verify.items():
            item_min_acc.append(min(v))

        acc = format(100 * np.mean(item_min_acc), ".4f")

    return acc, parsed_model_answers, correctness


@ray.remote(num_gpus=1)
class InferActor:
    def __init__(self, args):
        print("initializing model on GPU")
        self.llm = LLM(
            model=args.model_name_and_path,
            max_model_len=MAX_OUTPUT_TOKEN + 4096,
            limit_mm_per_prompt={"image": 10},
        )
        self.sampling_params = SamplingParams(
            temperature=0.4,
            max_tokens=MAX_OUTPUT_TOKEN,
            repetition_penalty=1.1,
        )
        # This sampling param is for the second "salvage" pass,
        # where we want a short, direct answer.
        self.salvage_params = SamplingParams(
            temperature=0.0,  # Be deterministic for the final answer
            max_tokens=256,  # Should be enough for just the \boxed{} part
            repetition_penalty=1.1,
        )

    def infer(self, llm_inputs, examples_subset, name, args):
        # --- Step 1: Main Generation Pass ---
        print(f"Performing main generation for {len(llm_inputs)} prompts...")
        initial_outputs = self.llm.generate(
            llm_inputs, sampling_params=self.sampling_params
        )

        final_responses = [None] * len(llm_inputs)
        prompts_to_salvage = []

        # --- Step 2: Identify Truncated Outputs and Prepare for Salvage ---
        for i, output in enumerate(initial_outputs):
            generated_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            # Check if truncated AND it looks like it's in a thought block
            if (
                finish_reason == "length"
                and "<think>" in generated_text
                and "</think>" not in generated_text
            ):

                # This prompt was truncated mid-thought. Prepare a salvage prompt.
                original_prompt = output.prompt

                # The new prompt is the original prompt + the truncated output + the closing tag
                # This gives the model the full context of its (partial) thought process.
                generated_text += (
                    "</think>" + "think is over, now give the **Final Answer**"
                )
                salvage_prompt = original_prompt + generated_text

                prompts_to_salvage.append(
                    {
                        "original_index": i,
                        "prompt": salvage_prompt,
                        "partial_response": generated_text,  # Store the first part
                    }
                )
            else:
                # This prompt finished correctly or was truncated outside a think block.
                # We consider it final.
                final_responses[i] = generated_text

        # --- Step 3: Run the Salvage Pass (if necessary) ---
        if prompts_to_salvage:
            print(f"Salvaging {len(prompts_to_salvage)} truncated outputs...")

            salvage_prompts_list = [p["prompt"] for p in prompts_to_salvage]
            salvage_outputs = self.llm.generate(
                salvage_prompts_list, self.salvage_params
            )

            for i, salvage_output in enumerate(salvage_outputs):
                item_to_salvage = prompts_to_salvage[i]
                original_index = item_to_salvage["original_index"]
                partial_response = item_to_salvage["partial_response"]

                # The final response is the original truncated part + the newly generated part
                salvaged_text = salvage_output.outputs[0].text
                final_responses[original_index] = partial_response + salvaged_text

        # --- Post-processing (same as before) ---

        # 解析结果并更新示例
        for i, example in enumerate(examples_subset):
            example.update({"resp": final_responses[i]})

        acc, model_answers, correctness = evaluate(examples_subset, name)

        # 保存单卡评估结果
        for example, answer, correct in zip(
            examples_subset, model_answers, correctness
        ):
            example.update({"parsed_answer": answer, "verify": correct})

        return {
            "name": name,
            "acc": acc,
            "examples": examples_subset,
        }


def main(args):
    # 取出模型名字
    path_parts = args.model_name_and_path.split(os.sep)
    model_name = (
        os.sep.join(path_parts[-3:])
        if len(path_parts) >= 3
        else args.model_name_and_path
    )

    # 1. 预加载处理器 / ray
    processor = AutoProcessor.from_pretrained(args.model_name_and_path)
    ray.init()

    benchmark_path = {
        # "MathVerse": os.path.join(
        #     ROOT, "MathVerse/testmini.json"
        # ),
        # "MathVista": os.path.join(ROOT, "MathVista/testmini.json"),  # 1,000
        # "LogicVista": os.path.join(ROOT, "LogicVista/data/dataset.json"),
        "MATH-V": os.path.join(ROOT, "MATH-V/data/test.jsonl"),  # 3040
        # "DynaMath": os.path.join(
        #     ROOT, "DynaMath/samples_and_result/combined_dataset.json",
        # ),
    }

    results = [["Dataset", "Acc"]]
    results_mean = []

    # 默认每多少道题保存一次
    save_batch_size = getattr(args, "save_batch_size", 128)

    # =================================================================
    #  核心改动：在这里创建 Actor，只创建一次！
    # =================================================================
    print("Creating inference actors...")
    num_gpus = int(ray.cluster_resources().get("GPU", 1))
    actors = [InferActor.options(num_gpus=1).remote(args) for _ in range(num_gpus)]
    print(f"{len(actors)} actors created and models are being loaded.")
    # 等待所有 actor 都初始化完毕（可选但推荐）
    ray.get(
        [actor.infer.remote([], [], "", args) for actor in actors]
    )  # 用一个空任务来触发初始化
    print("All actors are ready.")

    for name, path in benchmark_path.items():
        # ---------------- 2. 加载、预处理完整数据 ----------------
        data = load_data(path, name)
        llm_inputs, examples = preprocess_data(data, name, processor)

        # ---------------- 3. 计算批数 ----------------
        num_save_batches = math.ceil(len(llm_inputs) / save_batch_size)

        # 用于累加整个数据集的准确率
        total_correct = 0
        total_count = 0

        num_gpus = int(ray.cluster_resources().get("GPU", 1))
        actors = [InferActor.options(num_gpus=1).remote(args) for _ in range(num_gpus)]

        for save_idx in range(num_save_batches):
            # ------- 3.1 取当前 save_batch -------
            sb_start = save_idx * save_batch_size
            sb_end = min(sb_start + save_batch_size, len(llm_inputs))

            sb_inputs = llm_inputs[sb_start:sb_end]
            sb_examples = examples[sb_start:sb_end]

            # ------- 3.2 再按 GPU 数目拆分 infer_batch -------
            infer_bs = math.ceil(len(sb_inputs) / num_gpus)
            futures = []
            for gpu_id, actor in enumerate(actors):
                ib_start = gpu_id * infer_bs
                ib_end = min(ib_start + infer_bs, len(sb_inputs))
                if ib_start >= ib_end:
                    continue
                futures.append(
                    actor.infer.remote(
                        sb_inputs[ib_start:ib_end],
                        sb_examples[ib_start:ib_end],
                        name,
                        args,
                    )
                )
            gpu_results = ray.get(futures)

            batch_examples = []
            batch_acc = 0.0
            for gr in gpu_results:
                batch_examples.extend(gr["examples"])
                batch_acc += float(gr["acc"])

            # 取平均准确率（多 GPU）
            batch_acc /= max(1, len(gpu_results))

            # ------- 3.4 立即写盘 -------
            save_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name}.jsonl")
            with open(save_path, "a", encoding="utf-8") as f:
                for ex in batch_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(f"Batch {save_idx+1}/{num_save_batches} saved to {save_path}")
            # print(f"[{name}] 已保存 batch {save_idx+1}/{num_save_batches}, "
            #   f"size={len(batch_examples)}, acc={batch_acc:.4f}")

            # ------- 3.5 累加到数据集级别 -------
            total_correct += batch_acc * len(batch_examples)
            total_count += len(batch_examples)

        # ---------------- 4. 本数据集完成，计算全局准确率 ----------------
        dataset_acc = total_correct / total_count if total_count else 0.0
        results.append([name, f"{dataset_acc:.4f}"])
        results_mean.append(dataset_acc)

    # ---------------- 5. 所有数据集平均 ----------------
    if results_mean:
        mean_acc = float(np.mean(results_mean))
        results.append(["Mean", f"{mean_acc:.4f}"])

    print(tabulate(results, headers="firstrow", tablefmt="grid"))
    ray.shutdown()


# 辅助函数：数据加载
def load_data(path, name):
    if path.endswith("jsonl"):
        with open(path) as f:
            return [json.loads(line.strip()) for line in f]
    elif path.endswith("json"):
        with open(path, "r", encoding="utf-8") as f:
            # 读取JSON文件，确保返回列表（若JSON本身是列表则直接返回，若为单个对象则包裹为列表）
            data = json.load(f)
            return data
    else:
        return load_dataset(path)  # 假设存在load_dataset函数


# 辅助函数：数据预处理
def preprocess_data(data, name, processor):
    llm_inputs = []
    examples = []
    if name in ["DynaMath", "LogicVista", "MathVista"]:
        data = list(data.values())
    # import random

    # random.shuffle(data)
    if args.first_n:
        data = data[: int(args.first_n)]  # 如果指定了first_n参数，则只取前first_n个数据
    # process_example(data[0], name, processor)  # 预处理第一个示例以检查是否有错误
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_example, example, name, processor): example
            for example in data
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
            llm_input, example = future.result()
            llm_inputs.append(llm_input)
            examples.append(example)
    return llm_inputs, examples


# 辅助函数：结果保存
def save_results(examples, name, output_dir, model_path_3):
    model_path_split = model_path_3.split(os.sep)
    output_dir = os.path.join(output_dir, os.path.join(*model_path_split[:-1]))
    file_path = os.path.join(output_dir, f"{name}_{model_path_split[-1]}.json")
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {file_path}")
    with open(file_path, "a") as f:
        json.dump(examples, f, indent=4)


def eval_from_jsonl(filepath):
    """
    从JSONL文件读取数据并进行评估

    Args:
        filepath: JSONL文件路径

    Returns:
        评估结果，包括准确率、解析后的模型答案和正确性列表
    """
    # 读取JSONL文件内容
    all_preds = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                # 解析每一行的JSON对象
                item = json.loads(line.strip())
                all_preds.append(item)
    except FileNotFoundError:
        print(f"错误：文件 {filepath} 未找到")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
        return None, None, None
    except Exception as e:
        print(f"错误：读取文件时发生异常 - {e}")
        return None, None, None

    # 确定数据集名称（从文件路径中提取）
    # 这里假设文件名包含数据集名称，如"MathVerse.jsonl"
    filename = filepath.split("/")[-1]
    name = filename.split(".")[0]

    # 调用评估函数
    return evaluate(all_preds, name)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
