import logging
from dataclasses import dataclass
from typing import Callable
import jsonschema
from dataclasses_json import DataClassJsonMixin
import requests
from openai.types.chat import ChatCompletion
import traceback

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

import backoff

logger = logging.getLogger("aide")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False

# 清理传递的字典 反序列化回openai的对象
def clean_and_convert(response_dict):
    """清理并转换字典为 ChatCompletion 对象"""
    # 处理可能为 None 的 index 字段
    if 'choices' in response_dict:
        for i, choice in enumerate(response_dict['choices']):
            if choice.get('index') is None:
                choice['index'] = i
    
    return ChatCompletion.model_validate(response_dict)

# 与容器外通信传递Query Answer
def backoff_create_api(base_url,api_key,model_params):
    url = "http://10.35.136.75:8192//call_model_api"
    try:
        response = requests.post(url=url,json={"model_params":model_params},timeout=2700)
        completion = (response.json())["data"]
        chat_completion = clean_and_convert(completion)
    except Exception as e:
        logger.info(f"Backoff exception: {e}")
        logger.info(f"{traceback.format_exc()}")
        return False
    return chat_completion

def opt_messages_to_list(
    system_message: str | None,
    user_message: str | None,
    convert_system_to_user: bool = False,
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        if convert_system_to_user:
            messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }
