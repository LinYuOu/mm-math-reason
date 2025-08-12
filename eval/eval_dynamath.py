import json
from collections import defaultdict

total_samples = 501  # 总样本量


def calculate_accuracy(file_path):
    """计算单个模型文件的准确率"""
    image_key_counts = defaultdict(int)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # 筛选：仅保留有image字段且verify=true的记录
                if "image" not in data or data.get("verify") is not True:
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
            except:
                continue  # 忽略错误行

    # 筛选出"trial1到trial10均verify=true"的图片（次数=10）
    qualified_images = [k for k, v in image_key_counts.items() if v == 10]
    qualified_count = len(qualified_images)

    # 计算准确率
    accuracy = qualified_count / total_samples if total_samples != 0 else 0
    return accuracy, qualified_count


if __name__ == "__main__":
    # 定义所有模型的文件路径
    model_paths = {
        "3B_SFT_RL": "DynaMath.jsonl",
    }

    # 计算并输出每个模型的结果
    print(f"总样本量: {total_samples}\n")
    for model_name, file_path in model_paths.items():
        acc, count = calculate_accuracy(file_path)
        print(f"{model_name}:")
        print(f"  满足条件的图片数量: {count}")
        print(f"  准确率(acc): {acc:.6f}\n")
