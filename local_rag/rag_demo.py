from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingType,DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

load_dotenv()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=api_key)

documents = SimpleDirectoryReader("./assets").load_data()
# index = VectorStoreIndex.from_documents(documents)

embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key=api_key
)

index = VectorStoreIndex.from_documents(documents, embed_model=embedder)

query_engine = index.as_query_engine(llm=dashscope_llm)

response = query_engine.query("八戒向妇人要求把女儿嫁给他时，妇人对他说了什么？")

print(response)


