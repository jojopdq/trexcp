import nest_asyncio
import qdrant_client
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from core.proxy_model import ProxyModel
from raptor.base import RaptorPack, RaptorRetriever

nest_asyncio.apply()


class RaptorHandler:
    def __init__(self, config):
        client = qdrant_client.QdrantClient(
            host=config.database.host, port=config.database.port
        )
        self.client = client
        self.config = config
        self.rerank_processor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=config.rerank.top_n
        )
        self.collection_name = config.database.name + "_llama"
        self.index = None
        self.query_engine = None

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding.name,  # max_length=self.config.embedding.dim
        )

        self.llm = ProxyModel(
            model_name=self.config.llm.name,
            api_base=self.config.llm.api_base,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=2048,
        )
        self.raptor_pack = None

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def build(self, raw_data_path: str):
        documents = SimpleDirectoryReader(input_dir=raw_data_path).load_data()
        vector_store = QdrantVectorStore(
            client=self.client, collection_name=self.collection_name
        )
        self.raptor_pack = RaptorPack(
            documents,
            embed_model=self.embed_model,
            llm=self.llm,
            vector_store=vector_store,  # used for storage
            similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
            mode="collapsed",  # sets default mode
            transformations=[
                SentenceSplitter(chunk_size=400, chunk_overlap=50)
            ],  # transformations applied for ingestion
        )

    def ask(self, question: str):
        self.client = qdrant_client.AsyncQdrantClient(
            host=self.config.database.host, port=self.config.database.port
        )
        vector_store = QdrantVectorStore(
            aclient=self.client, collection_name=self.collection_name
        )
        retriever = RaptorRetriever(
            [],
            embed_model=self.embed_model,  # used for embedding clusters
            llm=self.llm,  # used for generating summaries
            vector_store=vector_store,  # used for storage
            similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
            mode="collapsed",  # sets default mode
            # mode="tree_traversal",  # sets default mode
        )
        query_engine = RetrieverQueryEngine.from_args(
            retriever,
            llm=self.llm,
        )
        return query_engine.query(question)

    def evaluate(self):
        pass
