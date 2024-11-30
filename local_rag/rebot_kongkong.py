from llama_index.core import PromptTemplate
from chat import rag

prompt_template = PromptTemplate(
    template="""
    You are an assistant who expert in Chinese fiction "Journey to the West" and named with "KongKong".
    Always answer in Chinese.
    The answer should be formatted in below format:
    "--------------------------------\n"
    "{context_str}\n"
    "--------------------------------\n"
    "Question:{query_str}\n"
    "--------------------------------\n"
    "KongKong: {answer_str}"
    """
)

index = rag.load_index()
query_engine = rag.create_query_engine(index)
query_engine.update_prompts({"response_synthesizer:text_qa_template":prompt_template})
rag.ask("八戒在濯垢泉发生了什么？",query_engine=query_engine,streaming=True,debug=False)
