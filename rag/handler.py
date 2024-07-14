import nest_asyncio
import qdrant_client
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from core import EvaluationMode, ProxyModel
from evaluator import DeepEvalEvaluator, RagEvaluatorPack

nest_asyncio.apply()


class RagHandler:
    def __init__(self, config):
        client = qdrant_client.QdrantClient(
            host=config.database.host, port=config.database.port
        )
        self.client = client
        self.aclient = qdrant_client.AsyncQdrantClient(
            host=config.database.host, port=config.database.port
        )
        self.config = config
        self.rerank_processor = SentenceTransformerRerank(
            model=config.rerank.name, top_n=config.rerank.top_n
        )
        self.collection_name = config.database.name + "_rag"
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

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

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

    def evaluate(self, evaluation_dataset_path: str, mode: EvaluationMode):
        match mode:
            case EvaluationMode.LlamaIndex:
                self.__evaluate_by_llama_index(evaluation_dataset_path)
            case EvaluationMode.DeepEval:
                self.__evaluate_by_deepeval(evaluation_dataset_path)

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

    def __evaluate_by_deepeval(self, evaluation_dataset_path: str):
        query_engine = self.__construct_query_engine()
        evaluator = DeepEvalEvaluator(query_engine=query_engine, llm=self.llm)
        dataset = evaluator.generate(evaluation_dataset_path=evaluation_dataset_path)
        evaluator.evaluate(dataset)

    def __evaluate_by_llama_index(self, evaluation_dataset_path: str):
        rag_dataset = LabelledRagDataset.from_json(evaluation_dataset_path)

        query_engine = self.__construct_query_engine()

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
