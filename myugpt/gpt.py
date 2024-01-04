"""
GPT COder to solve competitive programming problems
"""
from typing import List

import instructor
from openai import OpenAI

from myugpt.schema import CodingEnv, ModelPrediction

GPT_SYSTEM = """You are MyuGPT, an advanced coding chat bot that solves \
competitive programming problems. You will be provided with a problem \
statement (along with the previous list of steps taken to solve the problem, \
if applicable). You will produce the next step as python code to solve the \
problem."""


class MyuGPT:
    def __init__(self, max_history=10):
        self.client = instructor.patch(OpenAI())
        self.previous_messages = []
        self.max_history = max_history

    def step(
        self,
        env: CodingEnv,
        temperature: float = 0.9,
    ) -> ModelPrediction:
        """Step through the coding environment to solve the problem."""
        gpt_prediction = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_model=ModelPrediction,
            temperature=temperature,
            max_retries=2,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate JSON response",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": env.prompt,
                        },
                    ],
                },
            ],
            max_tokens=4096,
        )

        print(gpt_prediction)

        return gpt_prediction

    def sample(
        self,
        env: CodingEnv,
        temperature: float = 0.9,
        num_samples: int = 2,
    ) -> List[ModelPrediction]:
        """Sample the coding environment to solve the problem."""

        result = []

        for _ in range(num_samples):
            result.append(
                self.step(
                    env=env,
                    temperature=temperature,
                )
            )

        return result


if __name__ == "__main__":
    from myugpt.dataset import CodeContestsDataset

    dataset = CodeContestsDataset()
    frame = dataset[0]
    gpt = MyuGPT()

    print(type(frame))

    env = CodingEnv(
        dataset_frame=frame,
    )

    print(env.prompt)
    prediction = gpt.step(env)
