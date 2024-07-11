from .EmbeddingModels import BGEmbeddingModel
from .QAModels import GlmQAModel
from .RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from .SummarizationModels import GlmSummarizationModel


class RaptorHandler:

    def __init__(self, config, save_path: str):
        self.config = config
        self.ra_conf = RetrievalAugmentationConfig(
            summarization_model=GlmSummarizationModel(),
            qa_model=GlmQAModel(),
            embedding_model=BGEmbeddingModel(),
        )
        self.save_path = save_path
        self.client = None

    def build(self, raw_data_path: str):
        if not self.client:
            self.client = RetrievalAugmentation(
                config=self.ra_conf, tree=self.save_path
            )
        with open(raw_data_path, "r", encoding="ISO-8859-1") as file:
            text = file.read()
        self.client.add_documents(text)
        self.client.save(self.save_path)

    def ask(self, question: str):
        if not self.client:
            self.client = RetrievalAugmentation(
                config=self.ra_conf, tree=self.save_path
            )
        return self.client.answer_question(question=question)

    def evaluate(self):
        pass
