
from parallem.types import LLMResponse


def to_chat_completion_output(resp: LLMResponse):
    return {
        "id": resp._pr.response_id or "cached",
        "object": "chat.completion",
        "created": 1,
        "model": "unknown",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp.final_answer,
                    "refusal": None,
                    "finish_reason": "stop",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_token_details": {
                "cached_tokens": 0,
                "audio_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "service_tier": "default",
    }