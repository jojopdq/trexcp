import nest_asyncio
import qdrant_client
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from core import EvaluationMode, ProxyModel
from evaluator import DeepEvalEvaluator, RagEvaluatorPack
from raptor.base import RaptorPack, RaptorRetriever

nest_asyncio.apply()


class RaptorHandler:
    def __init__(self, config):
        self.client = qdrant_client.QdrantClient(
            host=config.database.host, port=config.database.port
        )
        self.aclient = qdrant_client.AsyncQdrantClient(
            host=config.database.host, port=config.database.port
        )
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
        vector_store = QdrantVectorStore(
            aclient=self.aclient, collection_name=self.collection_name
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

    def evaluate(self, evaluation_dataset_path: str, mode: EvaluationMode):
        match mode:
            case EvaluationMode.LlamaIndex:
                self.__evaluate_by_llama_index(evaluation_dataset_path)
            case EvaluationMode.DeepEval:
                self.__evaluate_by_deepeval(evaluation_dataset_path)

    def __evaluate_by_deepeval(self, evaluation_dataset_path: str):
        vector_store = QdrantVectorStore(
            aclient=self.aclient, collection_name=self.collection_name
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
        evaluator = DeepEvalEvaluator(query_engine=query_engine, llm=self.llm)
        dataset = evaluator.generate(evaluation_dataset_path=evaluation_dataset_path)
        evaluator.evaluate(dataset)

    def __evaluate_by_llama_index(self, evaluation_dataset_path: str):
        rag_dataset = LabelledRagDataset.from_json(evaluation_dataset_path)

        vector_store = QdrantVectorStore(
            aclient=self.aclient, collection_name=self.collection_name
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
        rag_evaluator_pack = RagEvaluatorPack(
            rag_dataset=rag_dataset,
            query_engine=query_engine,
            judge_llm=self.llm,
            embed_model=self.embed_model,
        )

        benchmark_df = rag_evaluator_pack.run(
            batch_size=20,  # batches the number of openai api calls to make
            sleep_time_in_seconds=1,
        )
        print(benchmark_df)
