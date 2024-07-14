import json
from typing import List

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from evaluator.deep_eval.llm import GLM4


class DeepEvalEvaluator:
    def __init__(self, query_engine, llm):
        self.query_engine = query_engine
        self.llm = GLM4(llm)

    def evaluate(self, dataset):

        faithfulnessMetric = FaithfulnessMetric(model=self.llm)
        contextualRecallMetric = ContextualRecallMetric(model=self.llm)
        contextualPrecisionMetric = ContextualPrecisionMetric(model=self.llm)
        hallucination_metric = HallucinationMetric(threshold=0.3, model=self.llm)
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.llm)

        evaluate(
            test_cases=dataset,
            metrics=[
                hallucination_metric,
                answer_relevancy_metric,
                faithfulnessMetric,
                contextualRecallMetric,
                contextualPrecisionMetric,
            ],
            ignore_errors=True,
        )

    def generate(self, evaluation_dataset_path: str):
        test_cases = []

        query_pairs = {}
        with open(evaluation_dataset_path) as json_file:
            data = json.load(json_file)
            examples = data["examples"]
            for example in examples:
                query_pairs[example["query"]] = example
        print(f"total questions:{len(query_pairs)}")
        for query in query_pairs.keys():
            response_object = self.query_engine.query(query)
            if response_object is not None:
                actual_output = response_object.response
                retrieval_context = [
                    node.get_content() for node in response_object.source_nodes
                ]
                source = query_pairs[query]
                test_case = LLMTestCase(
                    input=query,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context,
                    context=source["reference_contexts"],
                    expected_output=source["reference_answer"],
                )
                test_cases.append(test_case)
        dataset = EvaluationDataset(test_cases=test_cases)
        return dataset
