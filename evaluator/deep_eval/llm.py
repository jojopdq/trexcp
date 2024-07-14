from deepeval.models.base_model import DeepEvalBaseLLM

from core import ProxyModel


class GLM4(DeepEvalBaseLLM):
    def __init__(self, llm: ProxyModel):
        self.model = llm

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        generated_content = model.call_with_prompt(prompt)
        return generated_content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "GLM4"
