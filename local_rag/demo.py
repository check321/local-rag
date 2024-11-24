from dotenv import load_dotenv
import os
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core.base.llms.types import ChatMessage, MessageRole

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=api_key)

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="你是一个AI助手,你的名字叫大聪明，请用中文回答问题。"),
    ChatMessage(role=MessageRole.USER, content="你是谁？Steve Jobs是谁？")
]

responses = dashscope_llm.stream_chat(messages)
for response in responses:
    print(response.delta, end="")