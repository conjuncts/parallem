from types import SimpleNamespace
from typing import Dict, List

import parallem as pllm
from parallem.core.convert.completion_output import to_chat_completion_output
from parallem.core.convert.completions import from_chat_completion

def to_obj(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: to_obj(v) for k, v in data.items()})
    if isinstance(data, list):
        return [to_obj(item) for item in data]
    return data


# def completion(
#     model=LLM,
#     messages=messages,
#     max_completion_tokens=4096,
#     temperature=temperature,
#     seed=42,
#     top_p=0.9,
#     metadata=get_langfuse_metadata("metadata")
# ):
#     if _manager is None:
#         raise RuntimeError("No registered instance. Call register_instance() first.")

class Router:
    
    def __init__(
        self,
        path=".pllm/.litellm",
        cache_responses=True,
        **kwargs,
    ):
        self._manager: pllm.AgentOrchestrator = pllm.resume_directory(
            path,
            ignore_cache=not cache_responses,
            load_dotenv=True,
            strategy="batch",
        )


    async def acompletion(
        self,
        model: str, messages: List[Dict[str, str]], **kwargs
    ):
        to_native = [from_chat_completion(msg) for msg in messages]
        # print(to_native)
        # exit(0)
        with self._manager.agent("litellm_router") as agent:
            output = await agent.ask_llm(
                to_native,
                llm=model,
            )
            return to_obj(to_chat_completion_output(output))

    def _submit_and_close(self):
        self._manager.submit_and_close()

if __name__ == "__main__":
    import asyncio


    async def main():
        router = Router()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Please name an animal."},
                {"type": "text", "text": "It must be aquatic."},
            ]}
        ]

        
        output = await router.acompletion(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=messages,
        )
        print(output)

    asyncio.run(main())