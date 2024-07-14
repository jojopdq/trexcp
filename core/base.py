from enum import Enum


class EvaluationMode(str, Enum):
    LlamaIndex = "LlamaIndex"
    Deepeval = "Deepeval"
