import asyncio
import json
from typing import Any
import os
import re
import httpx

import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

async def eval_one_check(client: httpx.AsyncClient, user_prompt: str, args: dict) -> "tuple[bool, bool, bool]":
    """
    Returns:
        (judge_result, call_success, api_failed)
        - judge_result: True/False — judge's verdict (only meaningful if call_success and not api_failed)
        - call_success: True iff the API call succeeded AND the response parsed as the
          expected JSON schema. False indicates either an API-layer failure or a
          response-parse failure.
        - api_failed: True iff the API call itself raised (HTTPStatusError / timeout /
          connect error). Distinguishes "remote 502/timeout" from "judge returned
          malformed JSON" so upstream can MASK the trajectory instead of polluting
          GRPO advantage with a fake `score=0` signal.

    The legacy 2-tuple shape `(judge, call_success)` is preserved by always returning
    judge_result=False on any failure, so existing call sites that only use the first
    two elements continue to work; new code can read `api_failed` to distinguish.
    """

    sglang_model = args.get("sglang_model")
    sglang_url = args.get("sglang_url")
    temperature = args.get("temperature")
    top_p = args.get("top_p")
    max_new_tokens = args.get("max_new_tokens")
    max_tokens = args.get("max_tokens")
    retry_times = args.get("retry_times")

    payload = {
        "model": sglang_model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "title": "Checklist Evaluation Verdict",
                    "description": "Structured evaluation result for a single assistant turn against checklist criteria.",
                    "properties": {
                        "high_level_understanding_of_the_question": {
                            "type": "string",
                        },
                        "analysis_of_if_focus_on": {
                            "type": "string",
                        },
                        "analysis_of_pass_condition": {
                            "type": "string",
                        },
                        "analysis_of_failure_examples": {
                            "type": "string",
                        },
                        "answer": {
                            "type": "boolean",
                        }
                    },
                    "required": [
                        "high_level_understanding_of_the_question",
                        "analysis_of_if_focus_on",
                        "analysis_of_pass_condition",
                        "analysis_of_failure_examples",
                        "answer"
                    ],
                    "additionalProperties": False
                }
            }
        }
    }

    try:
        resp = await _post_with_retries(client, sglang_url, payload, retry_times)
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = json.loads(text)
            ans = parsed['answer']
            if ans not in [True, False]:
                raise ValueError("answer is not boolen")
            return ans, True, False
        except Exception as e:
            logger.warning(f"text can not be parsed in reward (call passed): {repr(e)}")
            return False, False, False
    except Exception as e:
        logger.warning(f"text can not be parsed in reward (call not passed): {repr(e)}")
        return False, False, True

def get_input_prompt_v2(messages_str_before_this_turn, messages_str_in_this_turn: str, this_turn_checklist: list[dict[str, Any]]) -> str:
    reference_snippet = [evidence['snippet'] for evidence in this_turn_checklist['evidence']]
    input_prompt = (
        "# Role\n"
        "You are a precise checklist evaluator. Your sole task is to judge whether the messages between user, assistant and tool satisfie the provided criteria.\n"
        "\n"
        "# Objective\n"
        "Produce a strict JSON verdict (no extra text) based on the instructions below.\n"
        "\n"
        "# Criteria\n"
        f"**Question:** {this_turn_checklist['question']}\n"
        f"**Focus on:** {this_turn_checklist['focus_on']}\n"
        f"**Pass condition:** {this_turn_checklist['pass_condition']}\n"
        f"**Failure examples:** {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=True, indent=2)}\n"
        f"**Reference snippet:** {json.dumps(reference_snippet, ensure_ascii=True, indent=2)}\n"
        "\n"
        "# Previous Messages\n"
        + messages_str_before_this_turn +
        "# Current Messages to Evaluate\n"
        + messages_str_in_this_turn +
        "\n"
        "# Special rule of tool call\n"
        "If there is no tool call in tool_call part but there are some tool calls in content.thinking part, it means these tools' format are not correct and all tool calls are not valid."
        "If there is error in tool response. The previous tool calls in latest assistant (only the latest one) are not valid."
        "# Evaluation Process (Align each step to a JSON output field)\n"
        "1. high_level_understanding_of_the_question:\n"
        "   - Briefly restate what is being evaluated (the intent of the question + what compliance means here).\n"
        "2. analysis_of_if_focus_on:\n"
        "   - Check whether Focus on part presents in the Current Messages.\n"
        "3. analysis_of_pass_condition:\n"
        "   - Determine if the 'Pass condition' is fully satisfied.\n"
        "4. analysis_of_failure_examples:\n"
        "   - For EACH failure example pattern: state clearly 'triggered' or 'not triggered' with a brief justification.\n"
        "5. answer:\n"
        "   - Return true ONLY IF:\n"
        "     * Focus on part is present.\n"
        "     * The 'Pass condition' is fully met.\n"
        "     * No failure example pattern is triggered.\n"
        "   - Otherwise return false.\n"
        "\n"
        "# Output Format\n"
        "Return ONLY a single JSON object with exactly these keys:\n"
        "{\n"
        "  \"high_level_understanding_of_the_question\": str,\n"
        "  \"analysis_of_if_focus_on\": str,\n"
        "  \"analysis_of_pass_condition\": str,\n"
        "  \"analysis_of_failure_examples\": str,\n"
        "  \"answer\": bool\n"
        "}"
    )

    return input_prompt

def get_messages_str_v2(messages: list[dict[str, Any]], step_num: int=None, max_length: int=40000) -> str:

    if step_num is not None and messages[0]["role"] == "assistant":
        assert len(messages) == 1, "Only one message is allowed when step_num is not None"
    turn = -1
    step = 0
    thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    messages_str = ""

    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            if_thinking = thinking_regex.search(content)
            if if_thinking:
                thinking = if_thinking.group(1)
                user_visible_reply = content.split(thinking+"</think>")[1].strip()
                if user_visible_reply == "":
                    user_visible_reply = "None"
            else:
                thinking = content
                user_visible_reply = "None"
            _tcs = message.get("tool_calls")
            if _tcs is not None:
                tool_calls = json.dumps(_tcs)
            elif "<tool_call>" in user_visible_reply or "</tool_call>" in user_visible_reply:
                tool_calls = user_visible_reply.replace("<tool_call>", "<|tool_call_start|>").replace("</tool_call>", "<|tool_call_end|>")
                user_visible_reply = "None"
            else:
                tool_calls = "None"
        
        if role == "system":
            messages_str += f"Role: system\ncontent: {content}\n"
            step = 0
        elif role == "user":
            turn += 1
            step = 0
            messages_str += f"# Turn: {turn}\nRole: user\ncontent: {content}\n"
        elif role == "assistant":
            if step_num is not None:
                _step = step_num
            else:
                _step = step
            this_step_message = f"## Step: {_step}\nRole: assistant\ncontent.thinking: {thinking}\ncontent.user_visible_reply: {user_visible_reply}\ntool_call: {tool_calls}\n"
            messages_str += this_step_message
            step += 1
        elif role == "observation" or role == "tool":
            messages_str += f"Role: tool\ncontent: {content}\n"
    return messages_str

async def _post_with_retries(client: httpx.AsyncClient, url: str, json_payload: dict, retry_times: int = 3) -> httpx.Response:
    """Post with retries and exponential backoff. Caller should handle failures.

    Backoff is exponential up to 30s per attempt (1s, 2s, 4s, 8s, 16s, ...),
    chosen so that 5 retries absorb transient 502/503/timeout from the upstream
    OpenAI-compatible gateway (typical recovery window 5-30s). The previous
    `(2**attempt)/10` capped at 1s was too short — we'd retry while the same
    upstream node was still unhealthy.
    """


    last_exc: Exception | None = None
    for attempt in range(max(1, int(retry_times))):
        try:
            resp = await client.post(url, json=json_payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = json.loads(text)
            return resp
        except Exception as e:
            last_exc = e
            try:
                await asyncio.sleep(min(2 ** attempt, 30))
            except Exception:
                pass
    assert last_exc is not None
    raise last_exc
