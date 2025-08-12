import asyncio
import regex as re
from typing import List
import random

import json
import math

from swift.plugin import ORM, orms
from swift.utils import get_logger

from math_verify import (
    StringExtractionConfig,
    ExprExtractionConfig,
    LatexExtractionConfig,
    LatexNormalizationConfig,
)
from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig

# from auto_scoring_judge import AutoScoringJudge

import base64
from openai import OpenAI
import concurrent.futures
import threading

logger = get_logger()


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util

        assert (
            importlib.util.find_spec("math_verify") is not None
        ), "The math_verify package is required but not installed. Please install it using 'pip install math_verify'."

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(gold_parsed, answer_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        rewards = []
        for content in completions:
            reward = 0.0

            if content.count("<think>") == 1:
                reward += 0.125

            if content.count("</think>") == 1:
                reward += 0.125

            if content.count("<answer>") == 1:
                reward += 0.125

            if content.count("</answer>") == 1:
                reward += 0.125

            rewards.append(reward)

        pattern = r"^<think>\n\n.*?\n\n</think>\n\n<answer>.*?</answer>(?![\s\S])"
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]

        return [
            reward + 0.5 if match else reward for reward, match in zip(rewards, matches)
        ]


class AnswerFormatORM(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = (
            r"^<desc>.*?</desc>\s*<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
        )
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if match else 0.0 for match in matches]


class TagCountFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        rewards = []
        for content in completions:
            reward = 0.0

            if content.count("<think>") == 1:
                reward += 0.25

            if content.count("</think>") == 1:
                reward += 0.25

            if content.count("<answer>") == 1:
                reward += 0.25

            if content.count("</answer>") == 1:
                reward += 0.25

            rewards.append(reward)

        return rewards


class BoxedCountFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        pattern = (
            r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
        )
        matches = [
            re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if len(match) == 1 else 0.0 for match in matches]


class BoxedFormatORM(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = (
            r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
        )
        matches = [
            re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if len(match) > 0 else 0.0 for match in matches]


class BoxedPosFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        return [self.boxed_distance_to_end(content) for content in completions]

    def boxed_distance_to_end(self, text):
        match = re.search(r"\\boxed", text)
        if match:
            position = match.start()
            distance = len(text) - position - len("\\boxed")
            return 1 - distance / len(text)
        else:
            return 0


class AccFormatORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0

            parse_content = parse(
                content,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            if not sol.startswith("$") and not sol.endswith("$"):
                sol = "$" + sol + "$"

            parse_sol = parse(
                sol,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            if verify(parse_sol, parse_content):
                reward = 1.0

            if reward == 0.0:
                format_pattern = r".+\n\n\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}(?![\s\S])"
                matches = re.match(format_pattern, content, re.DOTALL | re.MULTILINE)

                if not matches:
                    reward = -1.0

            rewards.append(reward)

        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify

        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(parse(sol), answer)) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                    ground_truth = (
                        sol_match.group(1).strip() if sol_match else sol.strip()
                    )

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r"<answer>(.*?)</answer>", content)
                    student_answer = (
                        content_match.group(1).strip()
                        if content_match
                        else content.strip()
                    )

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


class CompareAnsORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        answer_tag_pattern = (
            r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
        )

        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0

            content_answer_match = re.findall(
                answer_tag_pattern, content, re.DOTALL | re.MULTILINE
            )

            if content_answer_match:
                content_answer = content_answer_match[-1].strip()

                if compare_ans(content_answer, sol):
                    reward = 1.0
            rewards.append(reward)

        return rewards


def extract_boxed_content(content):
    """提取文本中\\boxed{}包裹的内容"""
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


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class AnswerTagAccuracyORM(ORM):
    # def __init__(self):
    #     self.scorer = AutoScoringJudge()

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for model_answer, sol in zip(completions, solution):
            reward = 0.0

            # model_answer_match = extract_boxed_content(model_answer)
            model_answer_match = re.search(r"<answer>(.*?)</answer>", model_answer)
            model_answer_match = (
                model_answer_match.group(1).strip()
                if model_answer_match
                else model_answer.strip()
            )

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
                model_answer = parse(
                    model_answer,
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                sol = parse(
                    fix_latex_answer(sol),
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                reward = float(verify(sol, model_answer))
            # reward = 1.0 if sol == 1 else 0.0
            rewards.append(reward)
        return rewards
        # return 1.0 if solution == 1 else 0.0


class AnswerTagAccuracyORM_Gauss(ORM):
    def __init__(self, numerical_tolerance: float = 1e-6, gaussian_sigma: float = 1.0):
        """
        Initializes the reward model with a Gaussian reward parameter.
        Args:
            numerical_tolerance (float): Tolerance for float and numeric SymPy comparisons.
            gaussian_sigma (float): Standard deviation for the Gaussian reward function.
                                    A larger sigma means more tolerance for numerical differences.
        """
        self.numerical_tolerance = numerical_tolerance
        self.gaussian_sigma = gaussian_sigma

    def __call__(
        self, completions: List[str], solution: List[str], **kwargs
    ) -> List[float]:
        rewards = []
        for model_answer_raw, sol_raw in zip(completions, solution):
            reward = 0.0  # Default reward is 0.0

            # Step 1: Extract model answer content (as per your original code)
            # model_answer_match_obj = re.search(r"<answer>(.*?)</answer>", model_answer_raw, re.DOTALL)
            model_answer_match_obj = extract_boxed_content(model_answer_raw)
            if isinstance(model_answer_match_obj, re.Match):
                model_answer_match_obj = model_answer_match_obj.group(1)
            model_answer_content = (
                model_answer_match_obj.strip()
                if model_answer_match_obj
                else model_answer_raw.strip()
            )

            # Check if extracted content is not empty
            if model_answer_content:
                try:
                    # Step 2: Apply fix_latex_answer and parse (as per your original code)
                    fixed_model_answer_str = fix_latex_answer(model_answer_content)
                    model_answer_parsed = parse(
                        fixed_model_answer_str,
                        extraction_config=[
                            StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                            ExprExtractionConfig(),
                            LatexExtractionConfig(),
                        ],
                    )

                    fixed_sol_str = fix_latex_answer(sol_raw)
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

                    if exact_match_reward == 1.0:
                        reward = 1.0  # Perfect match
                    else:
                        # Step 4: Apply Gaussian Reward if not exact match and both are numerically convertible
                        try:
                            model_float = float(model_answer_parsed)
                            sol_float = float(sol_parsed)

                            diff = model_float - sol_float
                            sigma_squared = self.gaussian_sigma**2
                            # Ensure sigma_squared is not zero
                            reward = math.exp(-(diff**2) / (2 * sigma_squared + 1e-9))

                        except (ValueError, TypeError, AttributeError):
                            # If conversion to float fails for either, it means they are not "pure numbers"
                            # for Gaussian reward. Reward remains 0.0 (or whatever default you prefer here).
                            reward = 0.0  # No Gaussian reward, no minimal reward

                except Exception:
                    # Catch any parsing or verification errors during this process
                    # Reward remains 0.0
                    reward = 0.0
            else:
                # If model_answer_content is empty (e.g., no <answer> tag and raw answer is empty)
                # Reward remains 0.0
                reward = 0.0

            rewards.append(reward)
        return rewards


class AnswerTagAccuracyLengthORM(ORM):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 512

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        for content, sol in zip(completions, solution):
            reward = 0.0

            content_match = re.search(r"<answer>(.*?)</answer>", content)
            content_answer = (
                content_match.group(1).strip() if content_match else content.strip()
            )

            content_answer = parse(
                content_answer,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            if not sol.startswith("$") and not sol.endswith("$"):
                sol = "$" + sol + "$"

            sol = parse(
                sol,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            completion_length = len(self.tokenizer.encode(content))
            coef = min(1, completion_length / self.max_length)

            reward = float(verify(sol, content_answer)) * coef

            rewards.append(reward)

        return rewards


class ThinkingVerifyORM(ORM):
    def __init__(self):
        base_url = "http://v2.open.venus.oa.com/llmproxy"
        token = "Lc5TeYAH3ijfGgGGGN0UF57b@1910"

        self.client = OpenAI(base_url=base_url, api_key=token)

        self.system_promopt = """You are an expert in evaluating reasoning processes. Your task is to determine whether the given reasoning process is correct, fluent, and complete. Please evaluate the reasoning process according to the following criteria and output the result:

1: If the reasoning process is completely correct, fluent, logically clear, and completely solves the problem.
0: If the reasoning process contains any errors, is not fluent, has logical leaps, omits steps, or fails to solve the problem correctly.

The final answer should be 0 or 1 based on above rules. Output analysis before your final answer and the final answer should enclosed within \\boxed{}"""

    def __call__(
        self, completions, solution, images, messages, **kwargs
    ) -> List[float]:
        rewards = []

        for content, sol, image, message in zip(
            completions, solution, images, messages
        ):
            reward = 0.0

            think_content_match = re.findall(
                r"<think>(.*?)</think>", content, re.DOTALL | re.MULTILINE
            )
            if len(think_content_match) == 1:
                think_content = think_content_match[0].strip()

                img_path = image[0]["path"]
                img_format = img_path.split(".")[-1]
                encoded_image = base64.b64encode(open(img_path, "rb").read()).decode()

                question = message[0]["content"].replace("<image>", "")
                try:
                    response = self.client.chat.completions.create(
                        model="gemini-2.0-flash",
                        messages=[
                            {"role": "system", "content": self.system_promopt},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/{img_format};base64,{encoded_image}"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f"Question: {question}\n\nReasoning: {think_content}",
                                    },
                                ],
                            },
                        ],
                        max_tokens=512,
                        temperature=0.0001,
                    )

                    resp = response.choices[0].message.content
                    # print(resp)
                    answer_pattern = r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
                    answer_match = re.findall(
                        answer_pattern, resp, re.DOTALL | re.MULTILINE
                    )

                    if answer_match:
                        gpt_answer = parse(
                            answer_match[-1],
                            extraction_config=[
                                StringExtractionConfig(
                                    strings=("A", "B", "C", "D", "E")
                                ),
                                ExprExtractionConfig(),
                                LatexExtractionConfig(),
                            ],
                        )

                        sol = parse(
                            sol,
                            extraction_config=[
                                StringExtractionConfig(
                                    strings=("A", "B", "C", "D", "E")
                                ),
                                ExprExtractionConfig(),
                                LatexExtractionConfig(),
                            ],
                        )

                        reward = float(verify(sol, gpt_answer))

                except Exception as e:
                    print(e)
                    pass

            rewards.append(reward)

        return rewards


# class CapVerifyORM(ORM):
#     def __init__(self):
#         base_url = 'http://v2.open.venus.oa.com/llmproxy'
#         token = "Lc5TeYAH3ijfGgGGGN0UF57b@1910"

#         self.client = OpenAI(base_url=base_url, api_key=token)

# #         self.system_promopt = '''You are an expert in describing image content. Your task is to determine whether the given caption is correct, fluent, and complete. Please evaluate the caption between 0 to 1 according to the following criteria and output the result:

# # 1 (highest score): If the caption is completely correct, fluent, logically clear, and considering every image detail.
# # 0 (lowest score): If the caption contains any errors, is not fluent, missing any key detailed information.

# # The final rating should enclosed within \\boxed{}'''

#     def __call__(self, completions, solution, images, messages, **kwargs) -> List[float]:
#         rewards = []

#         for content, sol, image, message in zip(completions, solution, images, messages):
#             reward = 0.0

#             cap_content_match = re.findall(r'<desc>(.*?)</desc>', content, re.DOTALL | re.MULTILINE)
#             if len(cap_content_match) == 1:
#                 cap_content = cap_content_match[0].strip()

#                 # img_path = image[0]["path"]
#                 # img_format = img_path.split(".")[-1]
#                 # encoded_image = base64.b64encode(open(img_path, "rb").read()).decode()

#                 question = message[0]["content"].replace("<image>", "")

#                 prompt = "Image: " + cap_content + "\n\nQuestion: " + question + "\n\nAnswer the question based on image, the final answer should be enclosed in \\boxed{}"
#                 # print(cap_content, question)
#                 try:
#                     response = self.client.chat.completions.create(
#                         model="gemini-2.0-flash",
#                         messages=[
#                             # {"role": "system", "content": self.system_promopt},
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     # {
#                                     #     "type": "image_url",
#                                     #     "image_url": {
#                                     #         "url": f"data:image/{img_format};base64,{encoded_image}"
#                                     #     }
#                                     # },
#                                     {
#                                         "type": "text",
#                                         "text": prompt
#                                     }
#                                 ]
#                             },

#                         ],
#                         max_tokens=1024,
#                         temperature=0.0001
#                     )

#                     resp = response.choices[0].message.content

#                     answer_pattern = r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
#                     answer_match = re.findall(answer_pattern, resp, re.DOTALL | re.MULTILINE)

#                     if answer_match:
#                         gpt_answer = parse(answer_match[-1], extraction_config=[
#                             StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
#                             ExprExtractionConfig(),
#                             LatexExtractionConfig(),
#                         ])

#                         sol = parse(sol, extraction_config=[
#                             StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
#                             ExprExtractionConfig(),
#                             LatexExtractionConfig(),
#                         ])

#                         reward = float(verify(sol, gpt_answer))

#                         # if reward == 1.0:
#                         #     print(resp)

#                 except Exception as e:
#                     print(e)
#                     pass

#             rewards.append(reward)

#         return rewards


class CapVerifyORM(ORM):
    def __init__(self):
        base_url = "http://v2.open.venus.oa.com/llmproxy"
        token = "Lc5TeYAH3ijfGgGGGN0UF57b@1910"

        self.client = OpenAI(base_url=base_url, api_key=token)

    def process_item(self, item_tuple):
        """Process a single item and return its reward."""
        idx, content, sol, image, message, question = item_tuple
        reward = 0.0

        # cap_content_match = re.findall(r'<desc>(.*?)</desc>', content, re.DOTALL | re.MULTILINE)
        # if len(cap_content_match) == 1:
        # cap_content = cap_content_match[0].strip()
        # question = message[0]["content"].replace("<image>", "")
        prompt = (
            "Image: "
            + content
            + "\n\nQuestion: "
            + question
            + "\n\nAnswer the question based on image, the final answer should be enclosed in \\boxed{}"
        )

        try:
            response = self.client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
                max_tokens=1024,
                temperature=0.0001,
            )

            resp = response.choices[0].message.content

            answer_pattern = (
                r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
            )
            answer_match = re.findall(answer_pattern, resp, re.DOTALL | re.MULTILINE)

            if answer_match:
                return (idx, answer_match[-1])

        except Exception as e:
            print(f"Thread processing error: {e}")

        return (idx, "")

    def __call__(
        self, completions, solution, images, messages, question, **kwargs
    ) -> List[float]:
        items_to_process = list(
            zip(
                range(len(completions)),
                completions,
                solution,
                images,
                messages,
                question,
            )
        )

        max_workers = min(32, len(items_to_process))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            answers = list(executor.map(self.process_item, items_to_process))

        answers = sorted(answers, key=lambda x: x[0])
        answers = [a[1] for a in answers]

        rewards = []
        for answer, sol in zip(answers, solution):

            parsed_answer = parse(
                answer,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            parsed_sol = parse(
                sol,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            rewards.append(float(verify(parsed_sol, parsed_answer)))

        return rewards


class BoxedVerifyORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        answer_tag_pattern = (
            r"\\boxed{((?:(?>[^{}]+)|{(?:(?>[^{}]+)|{(?:(?>[^{}]+)|{.*?})*})*})*)}"
        )

        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0

            parse_content = parse(
                content,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            # if not sol.startswith("$") and not sol.endswith("$"):
            #     sol = "$" + sol + "$"

            parse_sol = parse(
                sol,
                extraction_config=[
                    StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                    ExprExtractionConfig(),
                    LatexExtractionConfig(),
                ],
            )

            if verify(parse_sol, parse_content) or sol == content:
                reward += 0.5

            content_answer_match = re.findall(
                answer_tag_pattern, content, re.DOTALL | re.MULTILINE
            )

            if content_answer_match:
                content_answer = content_answer_match[-1].strip()

                parse_content_answer = parse(
                    content_answer,
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                if verify(parse_sol, parse_content_answer) or sol == content_answer:
                    reward += 0.5

            rewards.append(reward)

        return rewards


class CapLengthORM(ORM):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        for content, sol in zip(completions, solution):
            reward = 0

            cap_content_match = re.findall(
                r"<desc>(.*?)</desc>", content, re.DOTALL | re.MULTILINE
            )
            think_content_match = re.findall(
                r"<think>(.*?)</think>", content, re.DOTALL | re.MULTILINE
            )

            if len(cap_content_match) == 1 and len(think_content_match) == 1:
                cap_length = len(self.tokenizer.encode(cap_content_match[0]))
                think_length = len(self.tokenizer.encode(think_content_match[0]))

                reward = float(think_length > cap_length)

            rewards.append(reward)

        return rewards


class CapCheckORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        for content, sol in zip(completions, solution):
            reward = 0

            cap_content_match = re.findall(
                r"<desc>(.*?)</desc>", content, re.DOTALL | re.MULTILINE
            )
            # think_content_match = re.findall(r'<think>(.*?)</think>', content, re.DOTALL | re.MULTILINE)

            if len(cap_content_match) == 1:
                parsed_cap = parse(
                    cap_content_match[0],
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                parsed_sol = parse(
                    sol,
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                reward = 1 - float(verify(parsed_sol, parsed_cap))

            rewards.append(reward)

        return rewards


class MMORM(ORM):
    def __init__(self):
        self.cap_check_reward = CapCheckORM()
        self.cap_verify_reward = CapVerifyORM()

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = []

        for content, sol in zip(completions, solution):
            reward = 0.0

            content_match = re.search(
                r"<answer>(.*?)</answer>", content, re.DOTALL | re.MULTILINE
            )

            if content_match:
                content_answer = content_match.group(
                    1
                ).strip()  # if content_match else content.strip()
                content_answer = parse(
                    content_answer,
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                sol = parse(
                    sol,
                    extraction_config=[
                        StringExtractionConfig(strings=("A", "B", "C", "D", "E")),
                        ExprExtractionConfig(),
                        LatexExtractionConfig(),
                    ],
                )

                reward = float(verify(sol, content_answer))

            acc_rewards.append(reward)

        cap_check_rewards = self.cap_check_reward(completions, solution, **kwargs)
        cap_verify_rewards = self.cap_verify_reward(completions, solution, **kwargs)

        rewards = []

        # print(len(cap_check_rewards), len(cap_verify_rewards), len(rewards))
        for ar, ccr, cvr in zip(acc_rewards, cap_check_rewards, cap_verify_rewards):
            reward = 0.0

            if ar > 0:
                reward = ar + ccr + cvr

            rewards.append(reward)

        return rewards


orms["external_math_acc"] = MathAccuracy
orms["external_math_format"] = MathFormat
orms["external_r1v_acc"] = MultiModalAccuracyORM
orms["external_compare_acc"] = CompareAnsORM
orms["external_answer_format"] = AnswerFormatORM
orms["external_answer_tag_acc"] = AnswerTagAccuracyORM
orms["external_answer_tag_acc_gauss"] = AnswerTagAccuracyORM_Gauss
orms["soft_overlong"] = SoftOverlong
orms["external_boxed_verify"] = BoxedVerifyORM
orms["external_boxed_format"] = BoxedFormatORM
orms["external_tag_count"] = TagCountFormat
orms["external_boxed_count"] = BoxedCountFormat
orms["external_boxed_pos"] = BoxedPosFormat
orms["external_acc_format"] = AccFormatORM
orms["external_answer_tag_length_acc"] = AnswerTagAccuracyLengthORM
orms["external_thinking_verify"] = ThinkingVerifyORM
orms["external_cap_verify"] = CapVerifyORM
orms["external_cap_length"] = CapLengthORM
orms["external_cap_check"] = CapCheckORM
orms["external_mm"] = MMORM
