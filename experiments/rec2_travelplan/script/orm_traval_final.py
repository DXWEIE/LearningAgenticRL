# Copyright (c) ModelScope Contributors. All rights reserved.
# Outcome Reward Model (ORM) implementations for GRPO training.

import json
import os
from random import random
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from swift.infer_engine import InferRequest

if TYPE_CHECKING:
    from swift.megatron.arguments import MegatronArguments
    from swift.rlhf_trainers import GRPOConfig


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class MathAccuracy(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, accuracy_orm=None):
        super().__init__(args)
        self.min_len_value_wrong = args.cosine_min_len_value_wrong
        self.max_len_value_wrong = args.cosine_max_len_value_wrong
        self.min_len_value_correct = args.cosine_min_len_value_correct
        self.max_len_value_correct = args.cosine_max_len_value_correct
        self.max_len = args.cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        self.ngram_size = args.repetition_n_grams
        self.max_penalty = args.repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        assert args.soft_cache_length < args.soft_max_length
        self.soft_max_length = args.soft_max_length
        self.soft_cache_length = args.soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards



from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor

QWEN_KEY = 'sk-xxx'


JUDGE_PROMPT_BC_en = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
    - For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()


def call_openai(text_in, used_model="qwen-plus-2025-09-11", max_tries=10):
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for attempt in range(max_tries):
        try:
            chat_response = client.chat.completions.create(
                model=used_model,
                messages=[{"role": "user", "content": text_in}],
                temperature=0,
                top_p=0.95,
                timeout=60
            )
            content = chat_response.choices[0].message.content
            if content:
                return content
        except Exception as e:
            time.sleep(2+max_tries*3)
            if attempt == (max_tries - 1):
                print(f"LLM judge server error {e}")
                return None
            continue
    return None


# import re
# class DeepSearchReward(ORM):
#     """
#     调试专用：并发 LLM judge reward
#     """
#     def _get_single_reward(self, question, response, answer):
#         ## llm as judge
#         content_match = re.search(r'<answer>(.*?)</answer>', response)
#         student_answer = content_match.group(1).strip() if content_match else "no answer"
#         judge_prompt = JUDGE_PROMPT_BC_en.format(
#             question=question,
#             response=student_answer,
#             correct_answer=answer
#         )
#         llm_response = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
#         if llm_response is None:
#             return 0.1
        
#         llm_response_lower = llm_response.lower()
#         if 'incorrect' in llm_response_lower or 'b' in llm_response_lower:
#             return 0.0
#         elif 'correct' in llm_response_lower or 'a' in llm_response_lower:
#             return 1.0
#         else:
#             return 0.1

#     def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
#         batch_size = len(completions)
#         questions = kwargs.get('question')
#         # 控制并发数，例如设置为 5 或 10
#         max_workers = 2
        
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # 使用 list 保证顺序一致
#             tasks = [
#                 executor.submit(self._get_single_reward, q, c, s)
#                 for q, c, s in zip(questions, completions, solution)
#             ]
#             rewards = [task.result() for task in tasks]
#         print('rewards are ',rewards)
#         return rewards

import re
from typing import List
from concurrent.futures import ThreadPoolExecutor

# 保留你原有的依赖导入（无需修改）
# from xxx import ORM
# JUDGE_PROMPT_BC_en = ...
# def call_openai(...): ...

# class DeepSearchReward(ORM):
#     """
#     调试专用：并发 LLM judge + 工具调用格式校验 reward
#     新增规则：每步仅1个tool_call，必须紧跟tool_response；连续tool_call越多惩罚越重
#     """
#     # 🔥 修复：去掉staticmethod，改为实例方法，彻底解决参数报错
#     def check_tool_format_score(self, response: str) -> float:
#         """
#         核心格式校验：
#         1. 标签必须严格交替：tool_call → tool_response → tool_call → tool_response...
#         2. 禁止连续tool_call，连续越多惩罚越高
#         3. 禁止悬空tool_call（最后一个是tool_call无响应）
#         返回：格式合规分数 0~1
#         """
#         # 1. 提取所有工具标签的顺序
#         tag_pattern = re.compile(r'<(tool_call|tool_response)>', re.DOTALL)
#         tag_sequence = tag_pattern.findall(response)
        
#         if not tag_sequence:
#             return 1.0
        
#         max_consecutive_calls = 0
#         current_consecutive = 0
#         has_dangling_call = False
        
#         # 2. 遍历标签序列，统计连续tool_call
#         for i, tag in enumerate(tag_sequence):
#             if tag == "tool_call":
#                 current_consecutive += 1
#                 if current_consecutive > max_consecutive_calls:
#                     max_consecutive_calls = current_consecutive
#             else:
#                 current_consecutive = 0
        
#         # 3. 检查是否悬空（最后一个是tool_call，无响应）
#         if tag_sequence[-1] == "tool_call":
#             has_dangling_call = True
        
#         # 4. 按连续数量计算格式分
#         if max_consecutive_calls == 0:
#             format_score = 1.0
#         elif max_consecutive_calls == 1:
#             format_score = 0.8
#         elif max_consecutive_calls == 2:
#             format_score = 0.6
#         else:
#             format_score = 0.4
        
#         # 5. 悬空tool_call额外惩罚
#         if has_dangling_call:
#             format_score *= 0.7
        
#         return max(format_score, 0.0)

#     def _get_single_reward(self, question, response, answer):
#         """原有LLM答案正确性打分（无修改）"""
#         content_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
#         student_answer = content_match.group(1).strip() if content_match else "no answer"
#         judge_prompt = JUDGE_PROMPT_BC_en.format(
#             question=question,
#             response=student_answer,
#             correct_answer=answer
#         )
#         llm_response = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
#         if llm_response is None:
#             return 0.1
        
#         llm_response_lower = llm_response.lower()
#         if 'incorrect' in llm_response_lower or 'b' in llm_response_lower:
#             return 0.0
#         elif 'correct' in llm_response_lower or 'a' in llm_response_lower:
#             return 1.0
#         else:
#             return 0.1

#     def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
#         batch_size = len(completions)
#         questions = kwargs.get('question')
#         max_workers = 2
        
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # 1. 答案正确性得分
#             content_tasks = [
#                 executor.submit(self._get_single_reward, q, c, s)
#                 for q, c, s in zip(questions, completions, solution)
#             ]
#             content_rewards = [task.result() for task in content_tasks]
            
#             # 2. 格式得分（调用无报错）
#             format_rewards = [self.check_tool_format_score(resp) for resp in completions]
            
#             # 3. 最终得分
#             final_rewards = [
#                 max(content * fmt, 0.0) 
#                 for content, fmt in zip(content_rewards, format_rewards)
#             ]
        
#         print('答案正确性得分：', content_rewards)
#         print('格式合规性得分：', format_rewards)
#         print('最终总奖励：', final_rewards)
#         return final_rewards


import re
from typing import List
from concurrent.futures import ThreadPoolExecutor



class DeepSearchReward(ORM):
    """
    调试专用：并发 LLM judge + 工具调用格式校验 reward
    新增规则1：每步仅1个<reason>，如果后面不是<answer>，必须是一个<tool>，然后后面必须紧跟着<response>；连续<tool>越多惩罚越重
    新增规则2：<response>为工具调用次数，调用次数少于2次，奖励打折
    """
    # def check_tool_format_score(self, response: str) -> float:
    #     """
    #     核心格式校验：
    #     1. 标签必须严格交替：tool_call → tool_response → tool_call → tool_response...
    #     2. 禁止连续tool_call，连续越多惩罚越高
    #     3. 禁止悬空tool_call（最后一个是tool_call无响应）
    #     返回：格式合规分数 0~1
    #     """
    #     # 1. 提取所有工具标签的顺序
    #     tag_pattern = re.compile(r'<(tool|response)>', re.DOTALL)
    #     tag_sequence = tag_pattern.findall(response)
        
    #     if not tag_sequence:
    #         return 1.0
        
    #     max_consecutive_calls = 0
    #     current_consecutive = 0
    #     has_dangling_call = False
        
    #     # 2. 遍历标签序列，统计连续tool_call
    #     for i, tag in enumerate(tag_sequence):
    #         if tag == "tool":
    #             current_consecutive += 1
    #             if current_consecutive > max_consecutive_calls:
    #                 max_consecutive_calls = current_consecutive
    #         else:
    #             current_consecutive = 0
        
    #     # 3. 检查是否悬空（最后一个是tool_call，无响应）
    #     if tag_sequence[-1] == "tool":
    #         has_dangling_call = True
        
    #     # 4. 按连续数量计算格式分
    #     if max_consecutive_calls == 0:
    #         format_score = 1.0
    #     elif max_consecutive_calls == 1:
    #         format_score = 0.8
    #     elif max_consecutive_calls == 2:
    #         format_score = 0.3
    #     else:
    #         format_score = 0.2
        
    #     # 5. 悬空tool_call额外惩罚
    #     if has_dangling_call:
    #         format_score *= 0.5
        
    #     return max(format_score, 0.0)
    def check_full_structure_score(self, response: str) -> float:
        """
        针对【完整整段response】的最强格式校验：
        1. <tool> 总数 必须 == <response> 总数
        2. 每个 <tool> 前面必须有 <reason>
        3. 禁止连续 <tool>
        4. 禁止悬空 <tool>
        5. 标签顺序必须合法循环：reason → tool → response
        返回：格式分 0~1
        """
        # --------------------- 1. 提取所有标签按出现顺序 ---------------------
        pattern = re.compile(r'<(reason|tool|response|answer)>', re.DOTALL)
        all_tags = [match.group(1) for match in pattern.finditer(response)]

        if not all_tags:
            return 0.0

        # --------------------- 2. 统计数量 ---------------------
        total_reason = all_tags.count("reason")
        total_tool = all_tags.count("tool")
        total_response = all_tags.count("response")
        has_answer = "answer" in all_tags

        # --------------------- 3. 核心强规则：tool 与 response 必须一一对应 ---------------------
        

        # --------------------- 4. 每个 tool 前面必须有 reason ---------------------
        # 找到所有 tool 的位置
        tool_positions = [i for i, t in enumerate(all_tags) if t == "tool"]
        dis_cnt = 0
        for pos in tool_positions:
            if pos == 0 or all_tags[pos - 1] != "reason":
                dis_cnt += 1

        # --------------------- 5. 禁止连续 tool ---------------------
        max_consec = 0
        current_consec = 0
        for t in all_tags:
            if t == "tool":
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        dis_cnt += max_consec - 1  # 连续 tool 的惩罚
    
        if not has_answer: # 没有answer就没有分
            return 0
        
        if total_tool != total_response:
            dis_cnt += abs(total_tool - total_response)  # tool 与 response 数量不匹配的惩罚
        
         # --------------------- 6. 禁止悬空 tool ---------------------
        if all_tags and all_tags[-1] == "tool":
            return 0.0
        
        # --------------------- 7. 必须有 reason ---------------------
        if total_reason < total_tool:
            dis_cnt += abs(total_reason - total_tool)  # reason 与 tool 数量不匹配的惩罚

        
        # --------------------- 8. 最终格式分 ---------------------
        base_score = 1.0
        return max(base_score/dis_cnt, 0.0)

    # ===================== 新增：工具调用次数折扣打分方法 =====================
    def get_tool_count_score(self, response: str) -> float:
        """
        新增维度：工具调用次数奖励规则
        <response> 数量 = 实际工具调用次数
        次数 < 2 (0次/1次)：奖励打折；次数 ≥ 2：不打折
        返回：次数折扣系数 0~1（可自行调整折扣力度）
        """
        # 统计<response>标签的数量
        tool_response_num = len(re.findall(r'<response>', response))
        
        # 🔧 自定义折扣规则（可根据需求修改数值）
        if tool_response_num == 0:
            return 0.0  # 0次工具调用：打5折
        else:
            return min(tool_response_num/5,1.0)


    def _get_single_reward(self, question, response, answer):
        """原有LLM答案正确性打分（无修改）"""
        # 写入jsonl，question,response和answer
        with open('/data/coding/finetuned_model/final_grpo/judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
            json.dump({
                "question": question.replace('"','\"'),
                "response": response.replace('"','\"'),
                "answer": answer.replace('"','\"')
            }, f, ensure_ascii=False)
            f.write('\n')

        content_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else "no answer"
        judge_prompt = JUDGE_PROMPT_BC_en.format(
            question=question,
            response=student_answer,
            correct_answer=answer
        )
        llm_response = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
        if llm_response is None:
            return 0.1
        
        llm_response_lower = llm_response.lower()
        if 'incorrect' in llm_response_lower or 'b' in llm_response_lower:
            return 0.0
        elif 'correct' in llm_response_lower or 'a' in llm_response_lower:
            return 1.0
        else:
            return 0.1

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        batch_size = len(completions)
        questions = kwargs.get('question')
        max_workers = 2
        rollout_infos = kwargs.get('rollout_infos', {})
        # 从dict list里面获取dict的value
        num_turns = [rollout_info.get('num_turns', 1) for rollout_info in rollout_infos]
        print('num_turns:', num_turns)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 1. 答案正确性得分（原有）
            content_tasks = [
                executor.submit(self._get_single_reward, q, c, s)
                for q, c, s in zip(questions, completions, solution)
            ]
            content_rewards = [task.result() for task in content_tasks]
            
            # 2. 格式得分（原有）
            #format_rewards = [self.check_full_structure_score(resp) for resp in completions]

            # 4. 最终得分：2部分加
            final_rewards = [c*(min(t/5,1)) for c, t in zip(content_rewards, num_turns)]

        # 打印所有维度得分（方便调试）
        print('答案正确性得分：', content_rewards)
        #print('格式合规性得分：', format_rewards)
        #print('工具调用次数折扣得分：', tool_count_rewards)
        print('最终总奖励：', final_rewards)
        return final_rewards




open_traval_llm_judge_system_prompt = """你是一名深谙旅游行业、具有严谨逻辑与评测方法论的「旅行规划 LLM 代理综合评审员」。现需对同一用户 Query 下，LLM Agent A 与 Agent B 的回答结果（Answer）分别进行分维度量化评估，并最终给出综合得分与胜者。请严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
[用户原始提问]
</USER_QUERY>

<ANSWER_A>
[LLM Agent A 的完整回答]
</ANSWER_A>

<ANSWER_B>
[LLM Agent B 的完整回答]
</ANSWER_B>
——————————

二、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 匹配度（Relevance）：完整响应所有子需求/限制？顺序与场景贴合？
2. 可行性（Feasibility）：安排逻辑自洽、切实可行，避免明显冲突？
3. 细节丰富度（Details）：时间表、票价、交通耗时、Tips 等信息是否丰富且实用？
4. 清晰度（Clarity）：结构清晰、排版友好、可读性高？

【评分规则】
• 回答结果评测时需参考对应推理路径中的参考知识。
• 每个维度 0–10 分；0 表示“完全缺失”，10 表示“极为出色”。  
• 回答结果综合得分（overall_a）＝四个维度均值后四舍五入保留小数点后一位。 
—————————— 

四、综合得分与胜负判定

—————————— 
综合得分高的取胜，若综合得分相同，则胜负判定结果为 Tie。
—————————— 

【输出格式（严格遵循，不要添加多余内容）】
{
  "analysis": {,
    "answer_A": "<80-120 字中文评述：指出 A 答案亮点与不足>",
    "answer_B": "<80-120 字中文评述：指出 B 答案亮点与不足>"
    },
  "answer_scores": {
    "Agent_A": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
    },
    "Agent_B": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_b": <0-10>
    }
  },
  "winner": "<Agent_A | Agent_B | Tie>"
}
【重要要求】
• 先逐维度独立思考后再给分，确保公平客观。
• 所有评语仅基于提供的文本，不要引入外部信息。
• 所有中文评述需具体、可溯源（可引用原文片段或段落号）。
• 严格遵守 JSON 模板，以便后续程序解析。
• 评估应关注整条路线每个点是否都被覆盖到，在都覆盖了的前提下，再看路线信息的完整性，路线的合理性"""


import json
from typing import Tuple, Union
import pandas as pd
import statistics

def extract_traval_overall_scores(judge_result: Union[str, dict]) -> Tuple[float, float]:
    """
    从LLM评审结果中提取 Agent_A 和 Agent_B 的总分
    :param judge_result: 评审结果（dict 或 json字符串）
    :return: (overall_a, overall_b) 缺失则返回 0.0
    """
    # 1. 统一转换为字典
    try:
        if isinstance(judge_result, str):
            judge_result = json.loads(judge_result)
    except (json.JSONDecodeError, TypeError):
        return 0.0, 0.0

    # 2. 安全逐级提取分数（无键则返回0，避免报错）
    answer_scores = judge_result.get("answer_scores", {})
    
    agent_a = answer_scores.get("Agent_A", {})
    overall_a = float(agent_a.get("overall_a", 0))
    
    agent_b = answer_scores.get("Agent_B", {})
    overall_b = float(agent_b.get("overall_b", 0))

    return overall_a, overall_b


class OpenTravalReward(ORM):
    """
    调试专用：并发 LLM judge + 工具调用格式校验 reward
    新增规则1：每步仅1个<reason>，如果后面不是<answer>，必须是一个<tool>，然后后面必须紧跟着<response>；连续<tool>越多惩罚越重
    新增规则2：<response>为工具调用次数，调用次数少于2次，奖励打折
    """

    # def tournament_two_completion(self, comp1, comp2,user_input):
    #     judge_prompt1 = open_traval_llm_judge_system_prompt.replace('{用户原始提问}', user_input).replace('{LLM Agent A 的完整回答}', comp1).replace('{LLM Agent B 的完整回答}', comp2)
    #     llm_response1 = call_openai(judge_prompt1, used_model="qwen-plus-2025-09-11")
    #     try:
    #         overall_a1, overall_b1 = extract_traval_overall_scores(llm_response1)
    #     except Exception as e:
    #         print(f"Error extracting scores: {e}")
    #         overall_a1, overall_b1 = 0.0, 0.0
    #     judge_prompt2 = open_traval_llm_judge_system_prompt.replace('{用户原始提问}', user_input).replace('{LLM Agent A 的完整回答}', comp2).replace('{LLM Agent B 的完整回答}', comp1)
    #     llm_response2 = call_openai(judge_prompt2, used_model="qwen-plus-2025-09-11")
    #     try:
    #         overall_b2, overall_a2 = extract_traval_overall_scores(llm_response2) # 注意这里反过来
    #     except Exception as e:
    #         print(f"Error extracting scores: {e}")
    #         overall_a2, overall_b2 = 0.0, 0.0
        
    #     return overall_a1+overall_a2, overall_b1+overall_b2

    def tournament_two_completion(self, comp1, comp2, user_input):
        # 任务1：comp1 vs comp2
        def task1():
            judge_prompt = open_traval_llm_judge_system_prompt.replace('[用户原始提问]', user_input).replace('[LLM Agent A 的完整回答]', comp1).replace('[LLM Agent B 的完整回答]', comp2)
            try:
                resp = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
                return extract_traval_overall_scores(resp)
            except:
                return 0.0, 0.0
        # 任务2：comp2 vs comp1
        def task2():
            judge_prompt = open_traval_llm_judge_system_prompt.replace('[用户原始提问]', user_input).replace('[LLM Agent A 的完整回答]', comp2).replace('[LLM Agent B 的完整回答]', comp1)
            try:
                resp = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
                return extract_traval_overall_scores(resp)
            except:
                return 0.0, 0.0

        # 并发执行两个LLM调用
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(task1)
            f2 = executor.submit(task2)
            overall_a1, overall_b1 = f1.result()
            overall_b2, overall_a2 = f2.result()

        return overall_a1 + overall_a2, overall_b1 + overall_b2

    def __call__(self, completions: List[str], user_inputs: List[str], **kwargs) -> List[float]:
        group_size = len(completions)
        questions = kwargs.get('question')
        max_workers = 2
        rollout_infos = kwargs.get('rollout_infos', {})
        # 从dict list里面获取dict的value
        num_turns = [rollout_info.get('num_turns', 1) for rollout_info in rollout_infos]
        infer_requests = [rollout_info.get('infer_request_all', {}) for rollout_info in rollout_infos]
        print('num_turns:', num_turns)

        wins = [0.0] * group_size

        for i in range(group_size):
            for j in range(i+1, group_size):
                score_i, score_j = self.tournament_two_completion(completions[i], completions[j],user_inputs[0])
                if score_i > score_j:
                    wins[i] += 1
                elif score_i < score_j:
                    wins[j] += 1
                else: # ties
                    wins[i] += 0.5
                    wins[j] += 0.5

        ranks = pd.Series(wins).rank(method="min", ascending=False).tolist()
        max_rank = max(ranks)

        if max_rank == 1:
            group_rewards = [0.0] * group_size
        else:
            group_rewards = [(max_rank - r) / (max_rank - 1) for r in ranks]

        if len(group_rewards) > 1:
            mean_val = statistics.mean(group_rewards)
            # 样本标准差，和torch.std默认行为一致
            std_val = statistics.stdev(group_rewards) if len(group_rewards) > 1 else 0.0
        else:
            mean_val = 0.0
            std_val = 0.0

        # Z-score 标准化（和torch逻辑完全一样）
        normalized_rewards = [(x - mean_val) / (std_val + 1e-6) for x in group_rewards]
        # ============================================================================
        print('最终总奖励：', normalized_rewards)
        return normalized_rewards






open_traval_llm_judge_system_prompt_ori = """你是一名深谙旅游行业、具有严谨逻辑与评测方法论的「旅行规划 LLM 代理综合评审员」。现需对同一用户 Query 下，LLM Agent A 与 Agent B 的推理路径（Path）和回答结果（Answer）分别进行分维度量化评估，并最终给出综合得分与胜者。请严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
[用户原始提问]
</USER_QUERY>

<PATH_A>
[LLM Agent A 的完整推理路径]
</PATH_A>

<PATH_B>
[LLM Agent B 的完整推理路径]
</PATH_B>

<ANSWER_A>
[LLM Agent A 的完整回答]
</ANSWER_A>

<ANSWER_B>
[LLM Agent B 的完整回答]
</ANSWER_B>
——————————

二、推理路径评测（Path Evaluation）

——————————
【评估维度说明】
1. 推理广度（Breadth）：是否从多角度（时间、空间、交通、价格、政策等）全面覆盖问题，同时无冗余或重复步骤。
2. 需求匹配度（Relevance）：各步骤与用户核心需求契合程度。
3. 细节信息丰富度（Detail）：引用的事实、数据、时间点、费用、预约规则等细节是否充分、准确且有用。

【评分规则】
• 推理路径评测时要求只关注推理路径中的实际工具调用，不用关注推理内容对信息的深入分析。
• 每个维度 0–10 分；0 表示“完全缺失”，10 表示“极为出色”。  
• 推理路径综合得分（Overall_P）＝三个维度均值后四舍五入取整。  
——————————

三、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 匹配度（Relevance）：完整响应所有子需求/限制？顺序与场景贴合？
2. 可行性（Feasibility）：安排逻辑自洽、切实可行，避免明显冲突？
3. 细节丰富度（Details）：时间表、票价、交通耗时、Tips 等信息是否丰富且实用？
4. 清晰度（Clarity）：结构清晰、排版友好、可读性高？

【评分规则】
• 回答结果评测时需参考对应推理路径中的参考知识。
• 每个维度 0–10 分；0 表示“完全缺失”，10 表示“极为出色”。  
• 回答结果综合得分（Overall_A）＝四个维度均值后四舍五入取整。 
—————————— 

四、综合得分与胜负判定

—————————— 
综合得分 combined_scores = 0.6 * Overall_P（路径总体分） + 0.4 * Overall_A（答案总体分），四舍五入保留 1 位小数。
若 Combined 相同，则胜负判定结果为 Tie。
—————————— 

【输出格式（严格遵循，不要添加多余内容）】
{
  "analysis": {
    "path_A": "<80-120 字中文评述：指出 A 路径亮点与不足>",
    "path_B": "<80-120 字中文评述：指出 B 路径亮点与不足>",
    "answer_A": "<80-120 字中文评述：指出 A 答案亮点与不足>",
    "answer_B": "<80-120 字中文评述：指出 B 答案亮点与不足>"
    },
  "path_scores": {
    "Agent_A": {
      "breadth": <0-10>,
      "relevance": <0-10>,
      "detail": <0-10>,
      "overall_p": <0-10>
    },
    "Agent_B": {
      "breadth": <0-10>,
      "relevance": <0-10>,
      "detail": <0-10>,
      "overall_p": <0-10>
    }
  },
  "answer_scores": {
    "Agent_A": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
    },
    "Agent_B": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
    }
  },
  "combined_scores": {
    "Agent_A": <0-10>,
    "Agent_B": <0-10>
  },
  "winner": "<Agent_A | Agent_B | Tie>"
}
【重要要求】
• 先逐维度独立思考后再给分，确保公平客观。
• 所有评语仅基于提供的文本，不要引入外部信息。
• 所有中文评述需具体、可溯源（可引用原文片段或段落号）。
• 严格遵守 JSON 模板，以便后续程序解析。

【工具解释】
- search_poi工具用于在一个指定的城市内搜索兴趣点（POI）的地理空间信息。
- search_around工具通过设置圆心和半径，搜索圆形区域内的地点信息。
- web_search工具用于执行通用的、开放知识搜索。
- search_navigation工具除了起始点、终点经纬度，还可以设置waypoints途经点。因此针对多点路线导航，既可以通过多次调用不带waypoints的direction工具来完成规划，也可以通过调用单次带waypoints的direction工具来完成规划。因此评估应关注整条路线每个点是否都被覆盖到，在都覆盖了的前提下，再看路线信息的完整性，路线的合理性"""


import json
from typing import Union, Dict

def extract_travel_judge_full_scores_ori(judge_raw: Union[str, dict]) -> Dict[str, float | str]:
    """
    解析旅游评审JSON结构，安全提取所有核心分数与winner
    :param judge_raw: LLM输出的json字符串 或 已经load好的字典
    :return: 包含所有维度分数+winner的字典，缺失字段默认0.0 / ""
    """
    # 1. 统一转字典，解析失败兜底
    try:
        if isinstance(judge_raw, str):
            data = json.loads(judge_raw)
        else:
            data = judge_raw
    except Exception:
        return {
            "Agent_A_overall_p": 0.0,
            "Agent_B_overall_p": 0.0,
            "Agent_A_overall_a": 0.0,
            "Agent_B_overall_a": 0.0,
            "Agent_A_combined": 0.0,
            "Agent_B_combined": 0.0,
            "winner": ""
        }

    # 2. 逐层安全提取
    path_scores = data.get("path_scores", {})
    answer_scores = data.get("answer_scores", {})
    combined_scores = data.get("combined_scores", {})

    # 路径总分 overall_p
    a_path = path_scores.get("Agent_A", {})
    b_path = path_scores.get("Agent_B", {})
    a_overall_p = float(a_path.get("overall_p", 0.0))
    b_overall_p = float(b_path.get("overall_p", 0.0))

    # 答案总分 overall_a
    a_ans = answer_scores.get("Agent_A", {})
    b_ans = answer_scores.get("Agent_B", {})
    a_overall_a = float(a_ans.get("overall_a", 0.0))
    b_overall_a = float(b_ans.get("overall_a", 0.0))

    # 综合分
    a_comb = float(combined_scores.get("Agent_A", 0.0))
    b_comb = float(combined_scores.get("Agent_B", 0.0))

    # 获胜方
    winner = data.get("winner", "").strip()

    return {
        "Agent_A_overall_p": a_overall_p,
        "Agent_B_overall_p": b_overall_p,
        "Agent_A_overall_a": a_overall_a,
        "Agent_B_overall_a": b_overall_a,
        "Agent_A_combined": a_comb,
        "Agent_B_combined": b_comb,
        "winner": winner
    }


def messages_to_text(messages):
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n---\n"
    return text

class OpenTravalOriginalReward(ORM):
    """
    调试专用：并发 LLM judge + 工具调用格式校验 reward
    新增规则1：每步仅1个<reason>，如果后面不是<answer>，必须是一个<tool>，然后后面必须紧跟着<response>；连续<tool>越多惩罚越重
    新增规则2：<response>为工具调用次数，调用次数少于2次，奖励打折
    """

    def tournament_two_completion(self, comp1, comp2, user_input, traj1, traj2):
        infer_traj_1 = messages_to_text(traj1[2:])  # 跳过system和第一个user
        infer_traj_2 = messages_to_text(traj2[2:])  # 跳过system和第一个user

        content_match = re.search(r'<answer>(.*?)</answer>', comp1, re.DOTALL)
        answer1 = content_match.group(1).strip() if content_match else comp1

        content_match = re.search(r'<answer>(.*?)</answer>', comp2, re.DOTALL)
        answer2 = content_match.group(1).strip() if content_match else comp2

        # 任务1：comp1 vs comp2
        def task1():
            time.sleep(random.uniform(0.5, 8))
            judge_prompt = open_traval_llm_judge_system_prompt_ori.replace('[用户原始提问]', user_input).replace('[LLM Agent A 的完整回答]', answer1).replace('[LLM Agent B 的完整回答]', answer2).replace('[LLM Agent A 的完整推理路径]', infer_traj_1).replace('[LLM Agent B 的完整推理路径]', infer_traj_2)
            with open('/data/coding/travel_finetune/grpo_final/judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"prompt": judge_prompt}, ensure_ascii=False) + '\n')
            try:
                resp = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
                with open('/data/coding/travel_finetune/grpo_final/judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"response": resp}, ensure_ascii=False) + '\n')
                parsed_scores = extract_travel_judge_full_scores_ori(resp)
                return parsed_scores['Agent_A_combined'], parsed_scores['Agent_B_combined']
            except:
                return 0.0, 0.0
        # 任务2：comp2 vs comp1
        def task2():
            time.sleep(random.uniform(0.5, 8))
            judge_prompt = open_traval_llm_judge_system_prompt_ori.replace('[用户原始提问]', user_input).replace('[LLM Agent A 的完整回答]', answer2).replace('[LLM Agent B 的完整回答]', answer1).replace('[LLM Agent A 的完整推理路径]', infer_traj_2).replace('[LLM Agent B 的完整推理路径]', infer_traj_1)
            with open('/data/coding/travel_finetune/grpo_final/judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"prompt": judge_prompt}, ensure_ascii=False) + '\n')
            try:
                resp = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
                with open('/data/coding/travel_finetune/grpo_final/judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"response": resp}, ensure_ascii=False) + '\n')
                parsed_scores = extract_travel_judge_full_scores_ori(resp)
                return parsed_scores['Agent_A_combined'], parsed_scores['Agent_B_combined']
            except:
                return 0.0, 0.0

        # 并发执行两个LLM调用
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(task1)
            f2 = executor.submit(task2)
            overall_a1, overall_b1 = f1.result()
            overall_b2, overall_a2 = f2.result()

        return overall_a1 + overall_a2, overall_b1 + overall_b2

    def __call__(self, completions: List[str], user_inputs: List[str], **kwargs) -> List[float]:
        group_size = len(completions)
        rollout_infos = kwargs.get('rollout_infos', {})
        # 从dict list里面获取dict的value
        num_turns = [rollout_info.get('num_turns', 1) for rollout_info in rollout_infos]
        infer_requests = [rollout_info.get('infer_request_all', {}) for rollout_info in rollout_infos]
        infer_request_list = [infer_requests[i].get('messages', []) for i in range(len(infer_requests))]
        print('num_turns:', num_turns)
        print('len(completions):', len(completions))

        wins = [0.0] * group_size

        for i in range(group_size):
            for j in range(i+1, group_size):
                score_i, score_j = self.tournament_two_completion(completions[i], completions[j],user_inputs[0],infer_request_list[i],infer_request_list[j])
                if score_i > score_j:
                    wins[i] += 1
                elif score_i < score_j:
                    wins[j] += 1
                else: # ties
                    wins[i] += 0.5
                    wins[j] += 0.5

        ranks = pd.Series(wins).rank(method="min", ascending=False).tolist()
        max_rank = max(ranks)

        if max_rank == 1:
            group_rewards = [0.0] * group_size
        else:
            group_rewards = [(max_rank - r) / (max_rank - 1) for r in ranks]
        
        # 记录具体的rewards和winrate，便于后续排查，记录到jsonl
        with open('/data/coding/travel_finetune/grpo_final/reward_output.jsonl', 'a', encoding='utf-8') as f:
            json.dump({
                "wins": wins,
                "group_rewards": group_rewards,
                "num_turns": num_turns,
                "user_inputs": user_inputs[0]
            }, f, ensure_ascii=False)
            f.write('\n')


        # ============================================================================
        print('wins = ', wins)
        print('最终总奖励：', group_rewards)
        return group_rewards




open_traval_llm_judge_grpo_prompt = """你是一名深谙旅游行业、具有严谨逻辑与评测方法论的「旅行规划 LLM 代理综合评审员」。
现需对一用户 Query 下，LLM Agent 的推理路径（Path）和回答结果（Answer）分别进行分维度量化评估，并最终给出综合得分。请严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
[用户原始提问]
</USER_QUERY>

<PATH>
[LLM Agent 的完整推理路径]
</PATH>

<ANSWER>
[LLM Agent 的完整回答]
</ANSWER>

——————————

二、推理路径评测（Path Evaluation）

——————————
【评估维度说明】
1. 推理广度（Breadth）：是否从多角度（时间、空间、交通、价格、政策等）全面覆盖问题，同时无冗余或重复步骤。
2. 需求匹配度（Relevance）：各步骤与用户核心需求契合程度。
3. 细节信息丰富度（Detail）：引用的事实、数据、时间点、费用、预约规则等细节是否充分、准确且有用。

【评分规则】
• 推理路径评测时要求只关注推理路径中的实际工具调用，不用关注推理内容对信息的深入分析。
• 每个维度 0–10 分；0 表示“完全缺失”，10 表示“极为出色”。  
• 推理路径综合得分（Overall_P）＝三个维度均值后四舍五入取整。  
——————————

三、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 匹配度（Relevance）：完整响应所有子需求/限制？顺序与场景贴合？
2. 可行性（Feasibility）：安排逻辑自洽、切实可行，避免明显冲突？
3. 细节丰富度（Details）：时间表、票价、交通耗时、Tips 等信息是否丰富且实用？
4. 清晰度（Clarity）：结构清晰、排版友好、可读性高？

【评分规则】
• 回答结果评测时需参考对应推理路径中的参考知识。
• 每个维度 0–10 分；0 表示“完全缺失”，10 表示“极为出色”。  
• 回答结果综合得分（Overall_A）＝四个维度均值后四舍五入取整。 
—————————— 

四、综合得分计算

—————————— 
综合得分 combined_scores = 0.6 * Overall_P（路径总体分） + 0.4 * Overall_A（答案总体分），四舍五入保留 1 位小数。

—————————— 

【输出格式（严格遵循，不要添加多余内容）】
{
  "analysis": {
    "path": "<80-120 字中文评述：指出路径亮点与不足>",
    "answer": "<80-120 字中文评述：指出答案亮点与不足>"
    },
  "path_scores": {
    "Agent": {
      "breadth": <0-10>,
      "relevance": <0-10>,
      "detail": <0-10>,
      "overall_p": <0-10>
    }
  },
  "answer_scores": {
    "Agent": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
    }
  },
  "combined_scores": {
    "Agent": <0-10>
  },
}
【重要要求】
• 先逐维度独立思考后再给分，确保公平客观。
• 所有评语仅基于提供的文本，不要引入外部信息。
• 所有中文评述需具体、可溯源（可引用原文片段或段落号）。
• 严格遵守 JSON 模板，以便后续程序解析。

【工具解释】
- search_poi工具用于在一个指定的城市内搜索兴趣点（POI）的地理空间信息。
- search_around工具通过设置圆心和半径，搜索圆形区域内的地点信息。
- web_search工具用于执行通用的、开放知识搜索。
- search_navigation工具除了起始点、终点经纬度，还可以设置waypoints途经点。因此针对多点路线导航，既可以通过多次调用不带waypoints的direction工具来完成规划，也可以通过调用单次带waypoints的direction工具来完成规划。因此评估应关注整条路线每个点是否都被覆盖到，在都覆盖了的前提下，再看路线信息的完整性，路线的合理性"""



import json
from typing import Union, Dict

def extract_grpo_judge_scores(judge_raw: Union[str, dict]) -> Dict[str, float | str]:
    """
    解析最终评审JSON结构，安全提取所有评述、维度分数与综合分数
    :param judge_raw: LLM输出的json字符串 或 已经load好的字典
    :return: 包含所有维度分数+评述的字典，缺失字段默认0.0 / ""
    """
    # 1. 统一转字典，解析失败兜底返回默认值
    try:
        if isinstance(judge_raw, str):
            data = json.loads(judge_raw)
        else:
            data = judge_raw
    except Exception:
        return {
            "analysis_path": "",
            "analysis_answer": "",
            "agent_breadth": 0.0,
            "agent_relevance_path": 0.0,
            "agent_detail": 0.0,
            "agent_overall_p": 0.0,
            "agent_relevance_answer": 0.0,
            "agent_feasibility": 0.0,
            "agent_details": 0.0,
            "agent_clarity": 0.0,
            "agent_overall_a": 0.0,
            "agent_combined": 0.0
        }

    # 2. 逐层安全提取顶层节点
    analysis = data.get("analysis", {})
    path_scores = data.get("path_scores", {})
    answer_scores = data.get("answer_scores", {})
    combined_scores = data.get("combined_scores", {})

    # 3. 提取评述内容（路径+答案）
    analysis_path = analysis.get("path", "").strip()
    analysis_answer = analysis.get("answer", "").strip()

    # 4. 提取路径分数（path_scores -> Agent）
    agent_path = path_scores.get("Agent", {})
    agent_breadth = float(agent_path.get("breadth", 0.0))
    agent_relevance_path = float(agent_path.get("relevance", 0.0))
    agent_detail = float(agent_path.get("detail", 0.0))
    agent_overall_p = float(agent_path.get("overall_p", 0.0))

    # 5. 提取答案分数（answer_scores -> Agent）
    agent_answer = answer_scores.get("Agent", {})
    agent_relevance_answer = float(agent_answer.get("relevance", 0.0))
    agent_feasibility = float(agent_answer.get("feasibility", 0.0))
    agent_details = float(agent_answer.get("details", 0.0))
    agent_clarity = float(agent_answer.get("clarity", 0.0))
    agent_overall_a = float(agent_answer.get("overall_a", 0.0))

    # 6. 提取综合分数（combined_scores -> Agent）
    agent_combined = float(combined_scores.get("Agent", 0.0))

    # 7. 返回所有解析结果
    return {
        # 评述内容
        "analysis_path": analysis_path,
        "analysis_answer": analysis_answer,
        # 路径维度分数
        "agent_breadth": agent_breadth,
        "agent_relevance_path": agent_relevance_path,
        "agent_detail": agent_detail,
        "agent_overall_p": agent_overall_p,
        # 答案维度分数
        "agent_relevance_answer": agent_relevance_answer,
        "agent_feasibility": agent_feasibility,
        "agent_details": agent_details,
        "agent_clarity": agent_clarity,
        "agent_overall_a": agent_overall_a,
        # 最终综合分数
        "agent_combined": agent_combined
    }


class OpenTravalGrpoReward(ORM):
    """
    调试专用：并发 LLM judge + 工具调用格式校验 reward
    新增规则1：每步仅1个<reason>，如果后面不是<answer>，必须是一个<tool>，然后后面必须紧跟着<response>；连续<tool>越多惩罚越重
    新增规则2：<response>为工具调用次数，调用次数少于2次，奖励打折
    """

    def pure_llm_judge_reward(self, comp1, user_input, traj1):
        infer_traj_1 = messages_to_text(traj1[2:])  # 跳过system和第一个user

        content_match = re.search(r'<answer>(.*?)</answer>', comp1, re.DOTALL)
        answer1 = content_match.group(1).strip() if content_match else comp1

        if '<answer>' not in comp1 or '</answer>' not in comp1:
            return 0 # 没有answer标签，直接0分

        def task1():
            judge_prompt = open_traval_llm_judge_grpo_prompt.replace('[用户原始提问]', user_input).replace('[LLM Agent 的完整回答]', answer1).replace('[LLM Agent 的完整推理路径]', infer_traj_1)
            with open('/data/coding/travel_finetune/grpo_final/pure_grpo_judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"prompt": judge_prompt}, ensure_ascii=False) + '\n')
            try:
                resp = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
                with open('/data/coding/travel_finetune/grpo_final/pure_grpo_judge_debug_log.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"response": resp}, ensure_ascii=False) + '\n')
                parsed_scores = extract_grpo_judge_scores(resp)
                return parsed_scores['agent_combined']
            except:
                return 0.0
        
        # 并发执行两个LLM调用
        with ThreadPoolExecutor(max_workers=1) as executor:
            f1 = executor.submit(task1)
            overall_a1 = f1.result()

        return overall_a1

    def __call__(self, completions: List[str], user_inputs: List[str], **kwargs) -> List[float]:
        group_size = len(completions)
        rollout_infos = kwargs.get('rollout_infos', {})
        # 从dict list里面获取dict的value
        num_turns = [rollout_info.get('num_turns', 1) for rollout_info in rollout_infos]
        infer_requests = [rollout_info.get('infer_request_all', {}) for rollout_info in rollout_infos]
        infer_request_list = [infer_requests[i].get('messages', []) for i in range(len(infer_requests))]
        print('num_turns:', num_turns)
        print('len(completions):', len(completions))

        rewards = [0.0] * group_size
        wins = [0.0] * group_size
        for i in range(group_size):
            score_i = self.pure_llm_judge_reward(completions[i], user_inputs[0], infer_request_list[i])
            rewards[i] = score_i/10
            wins[i] = score_i
        
        # 记录具体的rewards和winrate，便于后续排查，记录到jsonl
        with open('/data/coding/travel_finetune/grpo_final/pure_llm_as_judge_reward_output.jsonl', 'a', encoding='utf-8') as f:
            json.dump({
                "wins": wins,
                "group_rewards": rewards,
                "num_turns": num_turns,
                "user_inputs": user_inputs[0]
            }, f, ensure_ascii=False)
            f.write('\n')

        # ============================================================================
        print('wins = ',wins)
        print('最终总奖励：', rewards)
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'deep_search_reward': DeepSearchReward,
    'open_traval_reward': OpenTravalReward,
    'open_traval_ori_reward': OpenTravalOriginalReward,
    'open_traval_grpo_reward': OpenTravalGrpoReward  # 非tournament的reward
}
