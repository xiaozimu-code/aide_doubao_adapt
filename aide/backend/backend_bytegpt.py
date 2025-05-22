"""Backend for OpenRouter(BYTE_GPT) API"""

import logging
import os
import time
import json
from funcy import notnone, once, select_values
import openai

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    backoff_create,
)

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_BYTE_GPT_client():
    global _client
    _client = openai.AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/v2/crawl",
        api_key=os.getenv("BYTE_GPT_API_KEY"),  
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_BYTE_GPT_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    print(f"filtered_kwargs:\n{filtered_kwargs}\n")
    # logger.info(f"log info filtered_kwargs:\n{filtered_kwargs}\n")
    # in case some backends dont support system roles, just convert everything to user
    messages = [
        {"role": "user", "content": message}
        for message in [system_message, user_message]
        if message
    ]

    t0 = time.time()
    logger.info(f"func_spec:{func_spec}")
    logger.info(f"----BYTE_GPT Querying----")
    # 应该是用于保证传入了一部分tools
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    logger.info(f"BYTE_GPT Query:\n{messages}\n")
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    choice = completion.choices[0]
    req_time = time.time() - t0
    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch:\nchoice.message.tool_calls[0]"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}\ntype:{type(choice.message.tool_calls[0].function.arguments)}"
            )
            raise e
    logger.info("BYTE_GPT Response:\n{output}")

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
