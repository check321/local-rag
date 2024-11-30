from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    load_index_from_storage)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.dashscope import (
    DashScope,
    DashScopeGenerationModels
    )
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from pathlib import Path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()
from typing import Optional

api_key = os.getenv("DASHSCOPE_API_KEY")


def create_index(doc_path: str = "./doca") -> VectorStoreIndex:
    """
    创建索引
    :param doc_path: 文档路径
    :return: 索引
    """
    if not Path(doc_path).exists:
        raise FileNotFoundError(f"File {doc_path} is not exist.")
    doc = SimpleDirectoryReader(doc_path).load_data()
    node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

    index = VectorStoreIndex.from_documents(
        doc,
        embed_model=DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
            api_key=api_key
        ),
        node_parser=node_parser,
    )

    return index


def indexing(doc_path: str = "./docs", persist_dir: str = "./kb/test"):
    """
    索引
    :param doc_path: 文档路径
    :param persist_dir: vec-data持久化路径
    """
    try:
        index = create_index(doc_path)
        index.storage_context.persist(persist_dir)
    except Exception as e:
        logging.error(f"Error indexing: {str(e)}")
        raise


def load_index(persist_dir: str = "./kb/test"):
    """
    加载索引
    :param persist_dir: vec-data持久化路径
    :return: 索引
    """

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context,embed_model=DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,api_key=api_key))

def create_query_engine(index: VectorStoreIndex):
    """
    创建查询引擎
    :param index: 索引
    :return: 查询引擎
    """
    query_engine = index.as_query_engine(streaming=True,llm=DashScope(model_name=DashScopeGenerationModels.QWEN_MAX,api_key=api_key))
    return query_engine

def debug_output(func):
    def wrapper(*args,**kwargs):
        result = func(*args,**kwargs)
        print('\n\nretrieve doc chunk:')
        for i, node in enumerate(result.source_nodes,1):
            print(f'[{i}] {node.text}')
        return result
    return wrapper

def ask(question: str,
        query_engine,
        streaming: bool = True,
        debug: bool = False
) -> Optional[str]:
    if not question.strip():
        raise ValueError("Say something first.")
    
    try:
        query_func = debug_output(query_engine.query) if debug else query_engine.query
        response = query_func(question)
        
        if streaming:
            response.print_response_stream()
            return None
        return str(response)
    except Exception as e:
        logging.error(f"Error querying: {str(e)}")
        raise

if __name__ == "__main__":
    # indexing()
    index = load_index()
    query_engine = create_query_engine(index)
    ask("用100字概括第71回说了什么内容",query_engine=query_engine,streaming=True,debug=False)
