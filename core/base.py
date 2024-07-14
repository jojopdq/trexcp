from enum import Enum


class EvaluationMode(str, Enum):
    LlamaIndex = "LlamaIndex"
    DeepEval = "DeepEval"
