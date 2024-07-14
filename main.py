import typer
import yaml
from easydict import EasyDict

from core import EvaluationMode
from rag import RagHandler
from raptor import RaptorHandler

app = typer.Typer()


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)


config = read_yaml_config("./config.yaml")


@app.command()
def build(mode: str = "rag", raw_data_path: str = "./data/source_files"):
    handler = fetch_handler(mode)
    handler.build(raw_data_path=raw_data_path)


@app.command()
def ask(mode: str = "rag", question: str = "什么样的数据算是个人数据"):
    handler = fetch_handler(mode)
    window_response = handler.ask(question)
    print(window_response)


@app.command()
def evaluate(
    mode: str = "rag",
    evaluation_dataset_path: str = "./data/evaluation_dataset/rag_dataset.json",
    evaluation_mode: EvaluationMode = EvaluationMode.DeepEval,
):
    handler = fetch_handler(mode)
    handler.evaluate(evaluation_dataset_path, evaluation_mode)


def fetch_handler(mode: str):
    handler = None
    match mode:
        case "rag":
            handler = RagHandler(config)
        case "raptor":
            handler = RaptorHandler(config)
        case _:
            pass
    if not handler:
        exit(1)
    return handler


if __name__ == "__main__":
    app()
