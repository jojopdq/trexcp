import qdrant_client
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from core.proxy_model import ProxyModel


class RagHandler:
    def __init__(self, config):
        client = qdrant_client.QdrantClient(
            host=config.database.host, port=config.database.port
        )
        self.client = client
        self.config = config
        self.rerank_processor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=config.rerank.top_n
        )
        self.collection_name = config.database.name + "_rag"
        self.index = None
        self.query_engine = None

        embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding.name,  # max_length=self.config.embedding.dim
        )

        llm = ProxyModel(
            model_name=self.config.llm.name,
            api_base=self.config.llm.api_base,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=2048,
        )

        Settings.llm = llm
        Settings.embed_model = embed_model

    def build(self, raw_data_path: str):
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        reader = SimpleDirectoryReader(input_dir=raw_data_path, recursive=True)
        documents = []
        for docs in reader.iter_data():
            for doc in docs:
                # do something with the doc
                print(doc.metadata)
                doc.text = doc.text.upper()
                documents.append(doc)

        nodes = node_parser.get_nodes_from_documents(documents)

        vector_store = QdrantVectorStore(
            client=self.client, collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex(nodes, storage_context=storage_context)

    def ask(self, question: str):
        query_engine = self.__construct_query_engine()
        response = query_engine.query(question)
        return response

    def evaluate(self, question: str):
        pass

    def __construct_query_engine(self):
        if not self.query_engine:
            index = self.__get_index()
            self.query_engine = index.as_query_engine(
                similarity_top_k=2,
                # the target key defaults to `window` to match the node_parser's default
                node_postprocessors=[
                    self.rerank_processor,
                    MetadataReplacementPostProcessor(target_metadata_key="window"),
                ],
            )
        return self.query_engine

    def __get_index(self):
        if not self.index:
            vector_store = QdrantVectorStore(
                client=self.client, collection_name=self.collection_name
            )
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
            )
        return self.index
