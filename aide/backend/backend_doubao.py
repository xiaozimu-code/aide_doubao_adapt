"""Backend for OpenRouter(DOUBAO) API"""

import logging
import os
import time
import json
from funcy import notnone, once, select_values
import openai
import pydantic

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    backoff_create,
    backoff_create_api,  # new api process
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
def _setup_doubao_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        api_key=os.getenv("DOUBAO_API_KEY"),
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_doubao_client()
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
    # print(f"----DOUBAO Querying----")
    logger.info(f"func_spec:{func_spec}")
    logger.info(f"----DOUBAO Querying----")
    logger.info(f"Doubao Query:\n{messages}\n")
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
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
    logger.info(f"Response choice:{choice}")
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
    logger.info(f"Doubao Response:\n{output}")
    # output = completion.choices[0].message.content
    # print(f"----DOUBAO Response:{output}\ntype:{type(output)}----")
    # logger.info(f"----DOUBAO Response:{output}\ntype:{type(output)}----")

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info


# new query to post api
def new_query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    # _setup_doubao_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    # print(f"filtered_kwargs:\n{filtered_kwargs}\n")
    # logger.info(f"log info filtered_kwargs:\n{filtered_kwargs}\n")
    # in case some backends dont support system roles, just convert everything to user
    messages = [
        {"role": "user", "content": message}
        for message in [system_message, user_message]
        if message
    ]

    t0 = time.time()
    # print(f"----DOUBAO Querying----")
    logger.info(f"func_spec:{func_spec}")
    logger.info(f"----DOUBAO Querying----")
    # logger.info(f"Doubao Query:\n{messages}\n")
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        
    logger.info(f"log info filtered_kwargs:\n{filtered_kwargs}\n")

    model_params={"messages":messages,"extra_body":{
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },**filtered_kwargs}
    
    logger.info(f"Doubao Query Params:\n{model_params}\n")

    completion = backoff_create_api(
        # base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        # api_key=os.getenv("DOUBAO_API_KEY"),
        model_params=model_params
    )   

    logger.info(f"api_response:\n{completion}\ntype:{type(completion)}")
    # choice = completion.choices[0]
    # logger.info(f"Response choice:{choice}")
    req_time = time.time() - t0
    if func_spec is None:
        # output = choice.message.content
        output = completion["content"]
    else:
        assert (
            # choice.message.tool_calls
            completion["tool_calls"]
        # ), f"function_call is empty, it is not a function call: {choice.message}"
        ), f"function_call is empty, it is not a function call: {completion}"

        assert (
            # choice.message.tool_calls[0].function.name == func_spec.name
            completion["tool_calls"][0]
        ), "Function name mismatch:\nchoice.message.tool_calls[0]"
        try:
            # output = json.loads(choice.message.tool_calls[0].function.arguments)
            output = json.loads(completion["tool_calls"][0].function.arguments)
        except json.JSONDecodeError as e:
            tool_calls = completion["tool_calls"][0].function.arguments
            tool_type = type(tool_calls)
            logger.error(
                # f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}\ntype:{type(choice.message.tool_calls[0].function.arguments)}"
                f"Error decoding the function arguments: {tool_calls}\ntype:{tool_type}"
            )
            raise e
    logger.info(f"Doubao Response:\n{output}")
    # output = completion.choices[0].message.content
    # print(f"----DOUBAO Response:{output}\ntype:{type(output)}----")
    # logger.info(f"----DOUBAO Response:{output}\ntype:{type(output)}----")

    # in_tokens = completion.usage.prompt_tokens
    # out_tokens = completion.usage.completion_tokens
    in_tokens = ""
    out_tokens = ""
    # info = {
    #     "system_fingerprint": completion.system_fingerprint,
    #     "model": completion.model,
    #     "created": completion.created,
    # }

    # return output, req_time, in_tok_count, out_tok_count, info
    return output, req_time

