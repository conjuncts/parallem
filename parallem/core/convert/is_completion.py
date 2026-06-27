from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
    from openai.types.chat.chat_completion_content_part_param import FileFile
    from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam
    from openai.types.chat.chat_completion_content_part_input_audio_param import ChatCompletionContentPartInputAudioParam
    from openai.types.chat.chat_completion_content_part_refusal_param import ChatCompletionContentPartRefusalParam

def is_chat_completion_part(
    obj: Union[
        dict,
        "ChatCompletionContentPartTextParam",
        # "ChatCompletionContentPartParam",
        "ChatCompletionContentPartImageParam",
        "ChatCompletionContentPartInputAudioParam",
        "FileFile",
        "ChatCompletionContentPartRefusalParam",
    ],
    role: str,
    exactly=False,
):
    """
    Check if an object exactly follows OpenAI's ChatCompletions API for a ContentPart.
    """

    valid_types = {
        "user": ["text", "image_url", "input_audio", "file"],
        "assistant": ["text", "refusal"],
        "system": ["text"],
        "developer": ["text"],
        "tool": ["text"],
    }
    
    if not isinstance(obj, dict):
        return False
    
    if role not in valid_types:
        return False

    if "type" not in obj:
        return False

    part_type = obj["type"]
    if part_type not in valid_types[role]:
        return False

    if part_type not in obj:
        # check that the key corresponding to the type is present
        # For example, if type is "text", then "text" key should be present
        return False

    content = obj[part_type]
    # lightly check types of content

    allowed_keys = {
        "image_url": {
            "url": (str, True),
            "detail": (str, False),
        },
        "input_audio": {
            "format": (str, True),
            "data": (str, True),
        },
        "file": {
            "file_data": (str, False),
            "file_id": (str, False),
            "filename": (str, True),
        }
    }
    if part_type in {"text", "refusal"}:
        if not isinstance(content, str):
            return False
    elif part_type in allowed_keys:
        if not isinstance(content, dict):
            return False
        for key, (expected_type, is_required) in allowed_keys[part_type].items():
            if is_required and key not in content:
                return False
            if key in content and not isinstance(content[key], expected_type):
                return False
        
        if exactly and any(key not in allowed_keys[part_type] for key in content):
            return False
    else:
        # unexpected part_type
        return False

    return True

def is_chat_completion(
    obj: Union[
        dict,
        "ChatCompletionMessageParam",
    ],
    exactly=False,
) -> bool:
    """
    Check if an object exactly follows OpenAI's ChatCompletions API.
    """

    if not isinstance(obj, dict):
        return False

    # check all required keys are present
    required_keys = ["role", "content"]
    if not all(key in obj for key in required_keys):
        return False

    # check no extra keys are present
    allowed_keys = {
        "developer": ["content", "role", "name"],
        "system": ["content", "role", "name"],
        "user": ["content", "role", "name"],
        "assistant": ["role", "content", "audio", "function_call", "name", "refusal", "tool_calls"],
        "tool": ["content", "role", "tool_call_id"],
        "function": ["content", "role", "name"],
    }

    role = obj["role"]

    if role not in allowed_keys:
        return False

    if exactly:
        for key in obj:
            if key not in allowed_keys[role]:
                return False

    # check content
    content = obj["content"]
    if not isinstance(content, (str, list)):
        return False

    if isinstance(content, list):
        for part in content:
            if not is_chat_completion_part(part, role, exactly=exactly):
                return False

    if role == "tool" and "tool_call_id" not in obj:
        return False

    return True
