# """
# ollama_client.py

# The one place that talks to Ollama. Every search asks two questions:
#   llm_parser.py  -> "which attributes?" (gender, colours, age, camera)
#   time_parser.py -> "which time?"
# Both send their question through ask_ollama() below, so the settings
# (temperature, thinking, timeout, ...) are the same for both and live here.
# """

# import json
# import logging
# import requests
# from langfuse import get_client
# from langfuse.openai import OpenAI
# from config import OLLAMA_THINK, OLLAMA_URL

# log = logging.getLogger("ollama")


# def ask_ollama(model, system_prompt, user_message, json_schema):
#     """
#     Sends one question to the model and returns two things:
#         answer   : the JSON the model wrote, as a dict, EXACTLY as written
#                    ({} if it wasn't valid JSON)
#         thinking : what the model thought before answering ("" if thinking is off)

#     system_prompt : the instructions, rules and examples (never changes)
#     user_message  : what the user typed (plus the calendar, for time)
#     json_schema   : the shape the answer must have (Ollama's "format" option)
#     """
#     body = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_message},
#         ],
#         # The answer must fit this shape (constrained decoding).
#         "format": json_schema,
#         # Let the model reason before writing the JSON (config.py).
#         "think": OLLAMA_THINK,
#         # Wait for the whole answer instead of receiving it word by word.
#         "stream": False,
#         # temperature 0 = always pick the most likely word,
#         # so the same question always gives the same JSON.
#         "options": {
#             "temperature": 0,
#             "num_ctx": 8192,
#         },
#         # Keep the model loaded for 30 minutes after a search, so the next
#         # search doesn't wait for it to load again (slow on a CPU).
#         "keep_alive": "30m",
#     }

#     try:
#         # The first search can be slow (loading the model on a CPU).
#         response = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300)
#     except requests.ConnectionError:
#         raise RuntimeError(f"Can't reach Ollama at {OLLAMA_URL}. Is Ollama running?")
#     except requests.Timeout:
#         raise RuntimeError(f"{model} took longer than 5 minutes to answer.")

#     if response.status_code != 200:
#         # For example: the model isn't downloaded ("ollama pull ..."), or
#         # OLLAMA_THINK = True with a model that can't think.
#         raise RuntimeError(f"Ollama returned an error: {response.text}")

#     reply = response.json()["message"]
#     answer = reply.get("content", "")
#     thinking = reply.get("thinking", "")
#     if thinking:
#         log.info("%s thought: %s", model, thinking)
#     log.info("%s answered: %s", model, answer)

#     try:
#         return json.loads(answer), thinking
#     except json.JSONDecodeError:
#         log.warning("%s's answer was not valid JSON: %s", model, answer)
#         return {}, thinking


import os
import json
import logging

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.100.63:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"

# --------------------------------------------------
# Initialize Langfuse + Ollama
# --------------------------------------------------

langfuse = get_client()

client = OpenAI(
    base_url=f"{OLLAMA_URL.rstrip('/')}/v1",
    api_key="ollama",
    timeout=300.0,
    max_retries=0,
)


# --------------------------------------------------
# Ask Ollama
# --------------------------------------------------


def ask_ollama(model, system_prompt, user_message, json_schema):
    """
    Send a request to Ollama through Langfuse OpenAI.

    Returns:
        answer: Parsed JSON as dict ({} if invalid)
        thinking: Model reasoning, if available
    """
    # On /v1, thinking is turned off with reasoning_effort "none".
    # ("think" is not a /v1 field, so Ollama ignores it.)
    extra_body = {"options": {"num_ctx": 8192}, "keep_alive": "30m"}
    if not OLLAMA_THINK:
        extra_body["reasoning_effort"] = "none"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": json_schema,
                    "strict": True,
                },
            },
            temperature=0,
            stream=False,
            top_p=1,
            extra_body=extra_body,
        )

    except APIConnectionError as exc:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_URL}. Is Ollama running?"
        ) from exc

    except APITimeoutError as exc:
        raise RuntimeError(f"{model} took longer than 5 minutes to answer.") from exc

    except APIStatusError as exc:
        raise RuntimeError(f"Ollama returned an error: {exc}") from exc

    reply = response.choices[0].message

    answer = reply.content or ""

    # Reasoning field may differ by Ollama version
    extra = reply.model_extra or {}

    thinking = (
        extra.get("reasoning")
        or extra.get("reasoning_content")
        or extra.get("thinking")
        or ""
    )

    if thinking:
        log.info("%s thought: %s", model, thinking)

    log.info("%s answered: %s", model, answer)

    try:
        return json.loads(answer), thinking

    except json.JSONDecodeError:
        log.warning(
            "%s's answer was not valid JSON: %s",
            model,
            answer,
        )
        return {}, thinking


# --------------------------------------------------
# Example usage
# --------------------------------------------------

# if __name__ == "__main__":
#     system_prompt = """
#     You are a helpful assistant.
#     Answer the user's question.
#     Always return valid JSON matching the schema.
#     """

#     user_message = "What is the capital of Pakistan?"

#     json_schema = {
#         "type": "object",
#         "properties": {
#             "answer": {"type": "string"},
#             "confidence": {"type": "number"},
#         },
#         "required": ["answer", "confidence"],
#         "additionalProperties": False,
#     }

#     try:
#         answer, thinking = ask_ollama(
#             model=OLLAMA_MODEL,
#             system_prompt=system_prompt,
#             user_message=user_message,
#             json_schema=json_schema,
#         )

#         print("\nANSWER:")
#         print(json.dumps(answer, indent=2))

#         print("\nTHINKING:")
#         print(thinking or "Not available")

#     finally:
#         # Ensure traces are delivered to Langfuse
#         langfuse.flush()
