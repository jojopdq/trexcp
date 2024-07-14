import yaml
from easydict import EasyDict

from core import EvaluationMode
from rag import RagHandler
from raptor import RaptorHandler


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)


config = read_yaml_config("./config.yaml")
# handler = RaptorHandler(config)
handler = RagHandler(config)


handler.build(raw_data_path="./data/source_files")

evaluation_dataset_path = "./data/evaluation_dataset/rag_dataset.json"
mode = EvaluationMode.DeepEval

handler.evaluate(evaluation_dataset_path, mode)

# async def run_evaluation() -> asyncio.coroutine:
#    await handler.evaluate(evaluation_dataset_path, mode)


# asyncio.run(run_evaluation())
