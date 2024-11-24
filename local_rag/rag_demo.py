from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,StorageContext
from dotenv import load_dotenv
import os
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingType,DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


load_dotenv()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=api_key)

# documents = SimpleDirectoryReader("./assets").load_data()

db = chromadb.PersistentClient("./chroma_db")
chroma_collection = db.get_or_create_collection("journey_to_west")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key=api_key
)
index = VectorStoreIndex.from_vector_store(vector_store,storage_context=storage_context,embed_model=embedder)

query_engine = index.as_query_engine(llm=dashscope_llm)

response = query_engine.query("八戒的人物性格是怎样的？有哪些例子？")

print(response)


