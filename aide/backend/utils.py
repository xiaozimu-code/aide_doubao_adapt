import logging
from dataclasses import dataclass
from typing import Callable
import jsonschema
from dataclasses_json import DataClassJsonMixin
import requests
from openai.types.chat import ChatCompletion
import traceback
import subprocess
import shlex
PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

import backoff
from enum import Enum
from pydantic import BaseModel, root_validator
from typing import Union, List, Dict, Any, Optional
import os

class AgentOutputStatus(str, Enum):
    NORMAL = "normal"
    CANCELLED = "cancelled"
    AGENT_CONTEXT_LIMIT = "agent context limit"


class AgentOutput(BaseModel):
    status: AgentOutputStatus = AgentOutputStatus.NORMAL
    content: Union[str, None] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # at least one of them should be not None
    @root_validator(pre=False, skip_on_failure=True)
    def post_validate(cls, instance: dict):
        assert (
                instance.get("status") is not AgentOutputStatus.NORMAL
                or instance.get("content") is not None
        ), "If status is NORMAL, content should not be None"
        return instance

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
    
    # return ChatCompletion.model_validate(response_dict)
    return AgentOutput.model_validate(response_dict)


# 与容器外通信传递Query Answer
def backoff_create_api(model_params):
     
    host_url = subprocess.check_output(
        ["sh", "-c", "ip route | awk '/default/ {print $3}'"]
    ).decode().strip()
    logger.info(f"get host url : {host_url}")
    forward_server_port = os.environ.get('FORWARD_PORT')
    url = f"http://{host_url}:{forward_server_port}/call_model_api"
    # completion = {"content":"completion手动初始化,接收到该值即模型调用失败！"}
    try:
        health_resp = requests.get(url = f"http://{host_url}:{forward_server_port}/health")
        logger.info(f"forward server status : {health_resp.text}")
        response = requests.post(url=url,json=model_params,timeout=5400)
        # logger.info(f"forward response:\n{response.text}")
        completion = response.json()
        if "content" not in completion:
            logger.info(f"本次模型接收结果异常:{completion}")
            completion = {"content":"completion手动初始化,接收到该值即模型调用异常！"}
        # chat_completion = clean_and_convert(completion)
        # chat_completion = clean_and_convert(response)
        if "error_message" in completion:
            logger.info(f"forward server status : {completion}")
    except Exception as e:
        logger.info(f"Backoff exception: {e}")
        logger.info(f"{traceback.format_exc()}")
        return False
    return completion

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
