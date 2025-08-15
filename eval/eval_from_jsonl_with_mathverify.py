"a file to evaluate with math verify"

import json
import os
import glob
import re
from collections import defaultdict
from eval_dynamath import calculate_accuracy
import argparse
from math_verify import (
    StringExtractionConfig,
    ExprExtractionConfig,
    LatexExtractionConfig,
)
from latex2sympy2_extended import NormalizationConfig
from math_verify import parse, verify

DYNAMATH_TOTAL_SAMPLES = 501  # 请根据实际情况修改这个数值


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


def extract_options(text, name):
    """提取选项内容"""
    # 定义正则表达式模式
    if name == "We-Math":
        # 匹配 A. content B. content 格式，直到遇到下一个选项或结尾
        pattern = r"([A-E])\.\s*(.*?)(?=[A-E]\.|$)"
    elif name == "MathVerse":
        # 修复：匹配 A:content B:content 格式，直到遇到下一个选项或结尾
        pattern = r"([A-E]):\s*(.*?)(?=[A-E]:|$)"
    else:
        # 默认尝试多种格式
        patterns = [
            r"([A-E]):\s*(.*?)(?=[A-E]:|$)",  # A:content B:content
            r"([A-E])\.\s*(.*?)(?=[A-E]\.|$)",  # A.content B.content
            r"\(([A-E])\)\s*(.*?)(?=\([A-E]\)|$)",  # (A) content (B) content
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches and len(matches) >= 2:
                results = {}
                for option, content in matches:
                    results[option] = content.strip()
                return results
        return {}

    # 使用正则表达式查找所有匹配项，添加 DOTALL 和 MULTILINE 标志
    matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)

    # 创建结果字典
    results = {}
    for option, content in matches:
        # 清理内容：去除首尾空白和多余的换行符
        cleaned_content = content.strip().replace("\n", " ").replace("\r", "")
        # 去除多余的空格
        cleaned_content = re.sub(r"\s+", " ", cleaned_content)
        results[option] = cleaned_content

    return results


def detect_dataset_name(filename):
    """根据文件名检测数据集类型"""
    filename_lower = filename.lower()
    if "mathverse" in filename_lower:
        return "MathVerse"
    elif "we-math" in filename_lower or "wemath" in filename_lower:
        return "We-Math"
    elif "mathvista" in filename_lower:
        return "MathVista"
    elif "dynamath" in filename_lower:
        return "DynaMath"
    else:
        return "Unknown"


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
        print(f"括号不匹配")
        return None  # 括号不匹配

    # 获取\boxed{}中的内容 (不包括花括号)
    boxed_content = content[start_pos : right - 1]

    return boxed_content


MATH_VERIFY_AVAILABLE = True


def math_verify_answer(model_ans, label_ans):
    """
    使用 math_verify 库验证数学答案
    如果库不可用或验证失败，返回 False
    """
    if not MATH_VERIFY_AVAILABLE:
        return False

    # Step 2: Apply fix_latex_answer and parse (as per your original code)
    fixed_model_answer_str = fix_latex_answer(model_ans)
    model_answer_parsed = parse(
        fixed_model_answer_str,
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
                    # try_extract_without_anchor=False,
                ),
            ],
        extraction_mode="first_match",
        )

    fixed_sol_str = fix_latex_answer(label_ans)
    sol_parsed = parse(
        fixed_sol_str,
        extraction_config=[
            StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ],
    )

    # Step 3: Initial Verification (as per your original code)
    # Assuming 'verify' returns 1.0 for true, 0.0 for false
    exact_match_reward = float(verify(sol_parsed, model_answer_parsed))
    return True if exact_match_reward else False


def normalize_choice_answer(
    answer, model_answer, choices=None, question=None, dataset_name=None
):
    """
    标准化选择题答案，使answer和model_answer格式一致进行比较

    Args:
        answer: true answer: 原始答案 (可能是文本或字母)
        model_answer: model 的答案 (可能是文本或字母)
        choices: 选项列表 (用于文本和字母之间的转换)
        question: 问题文本 (用于提取选项，当choices为None时)
        dataset_name: 数据集名称 (用于选择正确的解析模式)

    Returns:
        tuple: (normalized_answer, normalized_model_answer, is_match)
    """
    # 字母到索引的映射
    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}

    # 如果没有提供choices，尝试从question中提取
    options_dict = {}
    if not choices and question and dataset_name:
        options_dict = extract_options(question, dataset_name)
        if options_dict:
            # 将字典转换为列表，按字母顺序排列
            choices = [
                options_dict.get(letter, "") for letter in sorted(options_dict.keys())
            ]

    def get_choice_info(value):
        """获取选项的各种表示形式"""
        value_str = str(value).strip()

        # 如果是带符号的字母（如$A$）
        if value_str.startswith("$") and value_str.endswith("$") and len(value_str) == 3:
            letter = value_str[1].upper()
            if letter in letter_to_index:
                index = letter_to_index[letter]
                # 如果有选项列表，获取对应文本
                if choices and 0 <= index < len(choices):
                    text = choices[index]
                # 如果有选项字典，获取对应文本
                elif options_dict and letter in options_dict:
                    text = options_dict[letter]
                else:
                    text = letter
                return {"letter": letter, "index": index, "text": text}

        # 如果是单个字母（A, B, C, D等）
        if len(value_str) == 1 and value_str.upper() in letter_to_index:
            letter = value_str.upper()
            index = letter_to_index[letter]

            # 如果有选项列表，获取对应文本
            if choices and 0 <= index < len(choices):
                text = choices[index]
            # 如果有选项字典，获取对应文本
            elif options_dict and letter in options_dict:
                text = options_dict[letter]
            else:
                text = letter

            return {"letter": letter, "index": index, "text": text}

        # 如果是选项文本，尝试找到对应的字母
        if choices and value_str in choices:
            index = choices.index(value_str)
            letter = index_to_letter.get(index, str(index))
            return {"letter": letter, "index": index, "text": value_str}

        # 如果在选项字典中找到匹配的文本
        if options_dict:
            for letter, text in options_dict.items():
                if text == value_str:
                    index = letter_to_index.get(letter, -1)
                    return {"letter": letter, "index": index, "text": text}

        # 其他情况，保持原值
        return {"letter": value_str, "index": -1, "text": value_str}

    answer_info = get_choice_info(answer)
    model_info = get_choice_info(model_answer)

    # 比较逻辑：
    # 1. 如果两者都是有效字母选项，比较字母
    # 2. 如果两者都能解析为有效选项索引，比较索引
    # 3. 否则直接比较文本

    # 情况1：两者都是单字母格式（如 MathVerse ）
    if (
        len(str(answer).strip()) == 1
        and str(answer).strip().upper() in letter_to_index
        and len(str(model_answer).strip()) == 1
        and str(model_answer).strip().upper() in letter_to_index
    ):
        is_match = str(answer).strip().upper() == str(model_answer).strip().upper()
        return answer_info["letter"], model_info["letter"], is_match

    # 情况2：通过索引比较（如MathVista）
    elif answer_info["index"] >= 0 and model_info["index"] >= 0:
        is_match = answer_info["index"] == model_info["index"]
        return answer_info["letter"], model_info["letter"], is_match

    # 情况3：直接比较文本
    else:
        # print(f"Comparing text: {answer_info['text']} vs {model_info['text']}")
        is_match = answer_info["text"] == model_info["text"] or math_verify_answer(
            answer_info["text"], model_info["text"]
        )
        return answer_info["text"], model_info["text"], is_match


def load_jsonl(file_path):
    """自动识别并加载JSON或JSONL文件，返回条目列表"""
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            item = json.loads(line)
            items.append(item)
    print(f"JSON解析失败，已按JSONL格式加载文件，共 {len(items)} 条数据")
    return items


def process_dynamath_file(
    file_path,
    match_output_path,
    mismatch_output_path,
    total_samples=DYNAMATH_TOTAL_SAMPLES,
):
    """处理DynaMath文件，生成匹配和不匹配的文件，并返回统计信息"""
    image_key_counts = defaultdict(int)
    image_match_counts = defaultdict(int)
    all_records = []  # 存储所有verify=True的记录
    total_records_count = 0  # 总记录数
    verify_true_records_count = 0  # verify=True的记录数

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total_records_count += 1

            # 统计verify=True的记录
            if data.get("verify") is True:
                verify_true_records_count += 1

            if Eval_with_math_verify:
                pass
            else:
                # 筛选：仅保留有image字段且verify=true的记录
                if data.get("verify") is not True:
                    continue

            # 提取image编号（如"image20.png"中的20）
            image_path = data["image"]
            image_filename = image_path.split("/")[-1]  # 取"image20.png"部分
            if image_filename.startswith("image") and "." in image_filename:
                # 提取数字（去掉"image"前缀和".png"后缀）
                num_part = image_filename[len("image") : image_filename.rfind(".")]
                if num_part.isdigit():
                    image_key = int(num_part)
                    image_key_counts[image_key] += 1

                    # 检查answer和model_answer是否匹配
                    answer_str = fix_latex_answer(str(data.get("answer", "")))
                    parsed_answer_str = fix_latex_answer(
                        str(data.get("parsed_answer", ""))
                    )

                    # 使用math_verify或字符串比较
                    is_match = answer_str == parsed_answer_str or math_verify_answer(
                        answer_str, parsed_answer_str
                    )

                    if is_match:
                        image_match_counts[image_key] += 1

                    # 添加到记录列表
                    record = {
                        "question": data.get("question", ""),
                        "answer": data.get("answer", ""),
                        "parsed_answer": data.get("parsed_answer", ""),
                        "normalized_answer": answer_str,
                        "normalized_parsed_answer": parsed_answer_str,
                        "dataset_type": "DynaMath",
                        "is_match": is_match,
                        "image": data.get("image", ""),
                        "image_key": image_key,
                    }
                    all_records.append(record)

    # 计算verify=True的比例
    verify_true_ratio, _ = calculate_accuracy(file_path)

    # 筛选出"trial1到trial10均verify=true"的图片（次数=10）
    qualified_images = [k for k, v in image_key_counts.items() if v == 10]
    qualified_count = len(qualified_images)

    # 计算在qualified_images中，有多少图片的所有trial都有answer==parsed_answer
    match_qualified_images = [
        k for k in qualified_images if image_match_counts.get(k, 0) == 10
    ]
    match_ratio = (
        len(match_qualified_images) / qualified_count if qualified_count > 0 else 0
    )

    # 分离匹配和不匹配的记录
    matched_records = [r for r in all_records if r["is_match"]]
    mismatched_records = [r for r in all_records if not r["is_match"]]

    # 写入匹配文件
    with open(match_output_path, "w", encoding="utf-8") as f:
        for record in matched_records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    # 写入不匹配文件
    with open(mismatch_output_path, "w", encoding="utf-8") as f:
        for record in mismatched_records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"DynaMath详细统计:")
    print(f"  总记录数: {total_records_count}")
    print(f"  verify=True记录数: {verify_true_records_count}")
    print(f"  verify=True比例: {verify_true_ratio:.4f}")
    print(f"  总样本量: {total_samples}")
    print(f"  满足条件的图片数量: {qualified_count}")
    print(f"  其中answer==model_answer的图片数量: {len(match_qualified_images)}")
    print(f"  在合格图片中answer==model_answer的比例: {match_ratio:.4f}")
    print(f"  所有verify=True记录数: {len(all_records)}")
    print(f"  其中匹配记录数: {len(matched_records)}")
    print(f"  其中不匹配记录数: {len(mismatched_records)}")
    print(f"  匹配文件已保存到: {match_output_path}")
    print(f"  不匹配文件已保存到: {mismatch_output_path}")
    print(
        f"  图片统计详情: {dict(sorted(image_key_counts.items())[:10])}..."
    )  # 显示前10个

    return {
        "match_ratio": match_ratio,
        "match_count": len(matched_records),
        "mismatch_count": len(mismatched_records),
        "qualified_count": qualified_count,
        "total_records": len(all_records),
        "match_qualified_images": len(match_qualified_images),
        "total_file_records": total_records_count,
        "verify_true_records": verify_true_records_count,
        "verify_true_ratio": verify_true_ratio,
    }


def filter_and_process_standard(input_path, match_output_path, mismatch_output_path):
    """处理标准文件，生成匹配和不匹配两个文件，仅保留匹配比例计算"""
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 - {input_path}")
        return None

    filename = os.path.basename(input_path)
    print(f"正在处理标准文件: {filename}")

    # 检测数据集类型
    dataset_name = detect_dataset_name(filename)
    print(f"检测到数据集类型: {dataset_name}")

    # 加载数据
    items = load_jsonl(input_path)
    if not items:
        print("无有效数据可处理，跳过该文件")
        return None

    filtered_match = []  # 匹配记录
    filtered_mismatch = []  # 不匹配记录

    for item in items:
        # 检查必要字段
        if not all(field in item for field in ["answer", "parsed_answer", "question"]):
            continue

        # 标准化答案并比较
        _, _, is_match = normalize_choice_answer(
            item["answer"], item["parsed_answer"], item.get("choices"), item["question"], dataset_name
        )

        # 提取公共字段
        record = {
            # "question": item["question"],
            "answer": item["answer"],
            "parsed_answer": item["parsed_answer"],
            # "dataset_type": dataset_name,
            "is_match": is_match,
        }

        # 分离结果
        if is_match:
            filtered_match.append(record)
        else:
            filtered_mismatch.append(record)

    # 计算匹配比例
    match_ratio = len(filtered_match) / len(items) if len(items) > 0 else 0

    # 写入文件
    with open(match_output_path, "w", encoding="utf-8") as f:
        for item in filtered_match:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    with open(mismatch_output_path, "w", encoding="utf-8") as f:
        for item in filtered_mismatch:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"匹配比例: {match_ratio:.2%}")
    print(f"匹配记录数: {len(filtered_match)}")
    print(f"不匹配记录数: {len(filtered_mismatch)}")

    return {
        "match_ratio": match_ratio,
        "match_count": len(filtered_match),
        "mismatch_count": len(filtered_mismatch),
    }


def process_single_file(input_file, output_folder):
    """处理单个文件，根据文件名判断使用哪种处理方式"""
    filename = os.path.basename(input_file)

    print(f"\n{'='*50}")
    print(f"正在处理文件: {filename}")
    print(f"{'='*50}")

    # 判断是否为DynaMath文件
    if "DynaMath" in filename:
        # 使用DynaMath特殊处理
        match_output_filename = f"filtered_match_{filename}"
        mismatch_output_filename = f"filtered_mismatch_{filename}"
        match_output_file = os.path.join(output_folder, match_output_filename)
        mismatch_output_file = os.path.join(output_folder, mismatch_output_filename)

        result = process_dynamath_file(
            input_file, match_output_file, mismatch_output_file
        )

        if result:
            match_ratio = result["match_ratio"]
            verify_true_ratio = result["verify_true_ratio"]
            return {
                "filename": filename,
                "match_ratio": f"{match_ratio*100:.2f}",
                "mismatch_ratio": f"{(1-match_ratio)*100:.2f}",
                "verify_true_ratio": f"{verify_true_ratio*100:.2f}",
                "match_count": result["match_count"],
                "mismatch_count": result["mismatch_count"],
                "verify_true_count": result["verify_true_records"],
                "total_records_count": result["total_file_records"],
                "match_filtered_count": result["match_count"],
                "mismatch_filtered_count": result["mismatch_count"],
                "dataset_type": "DynaMath",
                "type": "DynaMath",
            }
        else:
            return None
    else:
        # 使用标准处理
        match_output_filename = f"filtered_match_{filename}"
        mismatch_output_filename = f"filtered_mismatch_{filename}"
        match_output_file = os.path.join(output_folder, match_output_filename)
        mismatch_output_file = os.path.join(output_folder, mismatch_output_filename)

        result = filter_and_process_standard(
            input_file, match_output_file, mismatch_output_file
        )
        if result:
            result["type"] = "Standard"
        return result


def process_all_files(input_folder, output_folder, file_pattern="*.jsonl"):
    """处理文件夹下的所有文件"""

    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹不存在 - {input_folder}")
        return

    # 查找所有匹配的文件
    search_pattern = os.path.join(input_folder, file_pattern)
    files = glob.glob(search_pattern)

    if not files:
        print(f"在文件夹 {input_folder} 中没有找到匹配 {file_pattern} 的文件")
        return

    print(f"找到 {len(files)} 个文件需要处理:")
    for file in files:
        print(f"  - {os.path.basename(file)}")

    # 存储所有结果
    all_match_results = {}
    all_mismatch_results = {}
    all_verify_true_results = {}
    detailed_results = {}

    # 处理每个文件
    for input_file in files:
        filename = os.path.basename(input_file)

        # 跳过特定文件（如果需要）
        if "filtered_" in filename or "We-Math" in filename:
            print(f"\n跳过文件: {filename} (包含过滤标识)")
            continue

        # 处理文件
        result = process_single_file(input_file, output_folder)
        if result:
            all_match_results[result["filename"]] = result["match_ratio"]
            all_mismatch_results[result["filename"]] = result["mismatch_ratio"]
            all_verify_true_results[result["filename"]] = result["verify_true_ratio"]
            detailed_results[result["filename"]] = result

    # 输出汇总结果
    print(f"\n{'='*60}")
    print("处理结果汇总:")
    print(f"{'='*60}")

    if all_match_results:
        print("匹配/不匹配/verify=True比例统计:")
        for filename, match_ratio in all_match_results.items():
            detail = detailed_results[filename]
            file_type = detail.get("type", "Unknown")
            dataset_type = detail.get("dataset_type", "Unknown")
            mismatch_ratio = detail.get("mismatch_ratio", "0.00")
            verify_true_ratio = detail.get("verify_true_ratio", "0.00")
            if filename == "We-Math":
                pass
            else:
                if Eval_with_math_verify:
                    print(f"  {filename:<30}:")
                    print(f"    [{file_type}|{dataset_type}] Match: {match_ratio}%")
                else:
                    print(f"  {filename:<30}:")
                    print(
                        f"    [{file_type}|{dataset_type}] Verify=True: {verify_true_ratio}% ({detail['verify_true_count']}/{detail['total_records_count']})"
                    )
                    print(
                        f"    [{file_type}|{dataset_type}] Match: {match_ratio}% ({detail['match_count']}/{detail['verify_true_count']})"
                    )
                    print(
                        f"    [{file_type}|{dataset_type}] Mismatch: {mismatch_ratio}% ({detail['mismatch_count']}/{detail['verify_true_count']})"
                    )
    else:
        print("没有成功处理任何文件")


# 主程序执行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSONL files with math verification."
    )
    parser.add_argument("--input_file", type=str,
                        help="Path to the input JSONL file.")
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the output folder for filtered results.",
    )
    args = parser.parse_args()

    # 用户指定了单个文件
    output_folder = (
        args.output_folder
        if args.output_folder
        else os.path.join(os.path.dirname(args.input_file), "filtered_results")
    )
    os.makedirs(output_folder, exist_ok=True)
    process_single_file(args.input_file, output_folder)
