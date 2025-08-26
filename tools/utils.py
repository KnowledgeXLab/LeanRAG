import json
import re
import subprocess
import threading
import time
import jieba
import os
import numpy as np
from openai import OpenAI
import requests
import tiktoken
import yaml

# Load config for API keys
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
DEEPSEEK_API_KEY = config['deepseek']['api_key']
print(f"DEEPSEEK_API_KEY: {DEEPSEEK_API_KEY}")
tokenizer = tiktoken.get_encoding("cl100k_base")
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


def create_if_not_exist(path):
    if not os.path.exists(path):  # 如果目录不存在，递归创建该目录
        os.makedirs(path, exist_ok=True)


def dicts_almost_equal(dict1, dict2, tolerance=1e-6):
    # 比较字典时允许浮动误差
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        # 如果值是列表，逐个元素比较
        if isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            for v1, v2 in zip(value1, value2):
                if isinstance(v1, float) and isinstance(v2, float):
                    if abs(v1 - v2) > tolerance:  # 浮动容差
                        return False
                elif v1 != v2:
                    return False
        # 如果值是浮点数，直接比较
        elif isinstance(value1, float) and isinstance(value2, float):
            if abs(value1 - value2) > tolerance:  # 浮动容差
                return False
        # 其他类型的值直接比较
        elif value1 != value2:
            return False
    return True


def custom_lower_fast(s):
    """中英文兼容的小写转换"""
    return s.lower() if s.isascii() else s  # 中文保持原样


def is_word_boundary(text, start, end):
    """自适应中英文词边界检测"""
    # 判断文本是否包含中文（包括扩展CJK字符）
    has_chinese = re.search(
        r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]", text)

    if has_chinese:
        # 中文模式：使用jieba分词检测词边界
        words = list(jieba.cut(text))
        current_pos = 0
        boundaries = set()

        # 构建词边界集合
        for word in words:
            boundaries.add(current_pos)  # 词开始位置
            boundaries.add(current_pos + len(word))  # 词结束位置
            current_pos += len(word)

        # 检查输入位置是否在分词边界上
        return start in boundaries or end in boundaries
    else:
        # 英文模式：使用正则表达式检测单词边界
        word_chars = r"\w"  # 仅字母、数字、下划线

        # 前字符检查
        prev_is_word = False
        if start > 0:
            prev_char = text[start - 1]
            prev_is_word = re.match(f"[{word_chars}]", prev_char, re.UNICODE)

        # 后字符检查
        next_is_word = False
        if end < len(text):
            next_char = text[end]
            next_is_word = re.match(f"[{word_chars}]", next_char, re.UNICODE)

        return not prev_is_word and not next_is_word


def read_jsonl(file_path):
    """
    读取jsonl文件或json数组文件，并返回包含JSON对象的列表。

    :param file_path: .jsonl或.json文件的路径
    :return: 包含JSON对象的列表
    """
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Try to parse as JSON array first
            if content.startswith('[') and content.endswith(']'):
                try:
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
                    pass

            # If not a JSON array, try parsing as JSONL
            f.seek(0)  # Reset file pointer
            for line in f:
                # 去除空行
                if line.strip():
                    json_obj = json.loads(line.strip())  # 解析每一行的JSON对象
                    data.append(json_obj)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def write_jsonl(data, path, mode="a", encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def write_jsonl_force(data, path, mode="w+", encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def check_test(entities):  # 初期用于检测是否有边界错误
    e_l = []
    for layer in entities:
        temp_e = []
        if type(layer) != list:
            temp_e.append(layer['entity_name'])
            e_l.append(temp_e)
            continue
        for item in layer:
            temp_e.append(item['entity_name'])
        e_l.append(temp_e)

    for index, layer in enumerate(entities):
        if type(layer) != list or len(layer) == 1:
            break
        for item in layer:
            if item['parent'] not in e_l[index+1]:
                print(item['entity_name'], item['parent'])


class InstanceManager:
    def __init__(self, url, ports, gpus, generate_model, startup_delay=30):
        self.ports = ports
        self.gpus = gpus
        self.base_url = url
        self.instances = []
        self.lock = threading.Lock()
        self.current_instance = 0  # 用于轮询策略
        self.generate_model = generate_model
        self.TOTAL_TOKEN_COST = 0
        self.TOTAL_API_CALL_COST = 0
        for port, gpu in zip(self.ports, self.gpus):
            self.instances.append({"port": port, "load": 0})

    def reset_token_cost(self):
        """重置总的token消耗和API调用次数"""
        self.TOTAL_TOKEN_COST = 0
        self.TOTAL_API_CALL_COST = 0

    def get_tokens_cosumption(self):

        return self.TOTAL_TOKEN_COST, self.TOTAL_API_CALL_COST

        # time.sleep(startup_delay)  # 等待所有实例启动

    def get_available_instance(self):
        """使用轮询策略获取一个可用的实例"""
        with self.lock:
            instance = self.instances[self.current_instance]
            self.current_instance = (
                self.current_instance + 1) % len(self.instances)
            return instance["port"]  # 返回端口

    def generate_text(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """发送请求到选择的实例"""
        port = self.get_available_instance()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------

        if len(history_messages) > 1:
            history_messages[0]['content'] = truncate_text(
                history_messages[0]['content'], max_tokens=3000)
            history_messages[1]['content'] = truncate_text(
                history_messages[1]['content'], max_tokens=25000)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        try:
            cur_token_cost = len(tokenizer.encode(messages[0]['content']))
            if cur_token_cost > 31000:
                cur_token_cost = 31000
                messages[0]['content'] = truncate_text(
                    messages[0]['content'], max_tokens=31000)

            # logging api call cost
            self.TOTAL_API_CALL_COST += 1

            # Handle OpenRouter API with OpenAI SDK
            if "openrouter.ai" in self.base_url:
                print(f"DEEPSEEK_API_KEY 2: {DEEPSEEK_API_KEY}")
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=DEEPSEEK_API_KEY,
                )

                # Remove incompatible parameters for OpenRouter
                clean_kwargs = {k: v for k, v in kwargs.items() if k not in [
                    "chat_template_kwargs"]}
                # print(f"clean_kwargs: {clean_kwargs}")
                # print(f"messages: {messages}")
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://github.com/RaZzzyz/LeanRAG",
                        "X-Title": "LeanRAG",
                    },
                    model=self.generate_model,
                    messages=messages,
                    **clean_kwargs
                )

                response_message = completion.choices[0].message.content
                if hasattr(completion, 'usage') and completion.usage:
                    self.TOTAL_TOKEN_COST += completion.usage.prompt_tokens
            else:
                # Original logic for other APIs
                base_url = f"{self.base_url}:{port}/v1"

                headers = {"Content-Type": "application/json"}
                request_data = {
                    "model": self.generate_model,
                    "messages": messages,
                    **kwargs,
                    "chat_template_kwargs": {"enable_thinking": False}
                }

                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=request_data,
                    timeout=240
                )

                response.raise_for_status()

                # Try to parse the response
                try:
                    if response.content:
                        res = json.loads(response.content)
                    else:
                        res = {}
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    res = {}

                # Extract the response message
                if res and "choices" in res and len(res["choices"]) > 0:
                    response_message = res["choices"][0]["message"]['content']
                    if "usage" in res:
                        self.TOTAL_TOKEN_COST += res["usage"]["prompt_tokens"]
                else:
                    response_message = ""

        except Exception as e:
            print(f"Retry for Error: {e}")
            response_message = ""
        print(
            f"Current TOTAL_TOKEN_COST: {self.TOTAL_TOKEN_COST}, TOTAL_API_CALL_COST: {self.TOTAL_API_CALL_COST}")
        print(f"Response message: {response_message}")
        return response_message

    async def generate_text_asy(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        """发送请求到选择的实例"""
        port = self.get_available_instance()
        base_url = f"{self.base_url}:{port}/v1"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get the cached response if having-------------------

        if len(history_messages) > 1:
            history_messages[0]['content'] = truncate_text(
                history_messages[0]['content'], max_tokens=3000)
            history_messages[1]['content'] = truncate_text(
                history_messages[1]['content'], max_tokens=25000)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        try:
            cur_token_cost = len(tokenizer.encode(messages[0]['content']))
            if cur_token_cost > 31000:
                cur_token_cost = 31000
                messages[0]['content'] = truncate_text(
                    messages[0]['content'], max_tokens=31000)

            # logging api call cost
            self.TOTAL_API_CALL_COST += 1
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": self.generate_model,
                    "messages": messages,
                    **kwargs,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
                timeout=240
            )
            response.raise_for_status()
            res = json.loads(response.content)
            self.TOTAL_TOKEN_COST += res["usage"]["prompt_tokens"]
            # 对结果进行后处理
            response_message = res["choices"][0]["message"]['content']
        except Exception as e:
            print(f"Retry for Error: {e}")
            response = ""
            response_message = ""

        return response_message
