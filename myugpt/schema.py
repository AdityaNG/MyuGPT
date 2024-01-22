"""
Models
"""
import ast
from typing import List

from instructor import llm_validator
from pydantic import BaseModel, BeforeValidator, validator
from typing_extensions import Annotated

from myugpt.helper import text_similarity


class ProgramInputs(BaseModel):
    """Program Input"""

    data: List[str]


class ProgramOutputs(BaseModel):
    """Program Output"""

    data: List[str]


class DatasetFrame(BaseModel):
    """Dataset Frame"""

    problem_statement: str
    inputs: ProgramInputs
    expected_outputs: ProgramOutputs

    def __str__(self):
        rep = "Problem Statement: " + self.problem_statement + "\n"
        rep += "=============\n"
        for index, (inp, out) in enumerate(
            zip(self.inputs.data, self.expected_outputs.data)
        ):
            rep += f"Input[{index}]:\n" + inp + "\n"
            rep += f"ExpectedOutputs[{index}]:\n" + out + "\n"
            rep += "=============\n"

        return rep


class PythonCode(BaseModel):
    """Python Code"""

    data: Annotated[
        str, BeforeValidator(llm_validator("Write a valid Python code"))
    ]

    @validator("data")
    def is_valid_python(cls, v):
        try:
            ast.parse(v)
        except SyntaxError as se:
            raise ValueError(
                "Invalid Python code:\n",
                v,
                "SyntaxError:",
                se.msg,
                "at line",
                se.lineno,
            )
        return v


class ModelPrediction(BaseModel):
    """Model Prediction"""

    thought_process: Annotated[
        str,
        BeforeValidator(
            llm_validator(
                "Explain your thought process for solving the problem"
            )
        ),
    ]
    code: PythonCode
    # predicted_outputs: Annotated[
    #     ProgramOutputs,
    #     BeforeValidator(
    #         llm_validator(
    #             "What do you think the outputs of your code will be?"
    #         )
    #     ),
    # ]
    score: Annotated[
        float,
        BeforeValidator(
            llm_validator("Score the correctness of your code (0 to 100)")
        ),
    ]

    # @property
    # def score(self) -> float:
    #     return self._score

    # @score.setter
    # def score(self, value: float):
    #     if value < 0 or value > 100:
    #         raise ValueError("Score must be between 0 and 100")
    #     self._score = value

    def __str__(self):
        rep = "ThoughtProcess:\n" + self.thought_process + "\n"
        rep += "Code:\n" + self.code.data + "\n"
        rep += "=============\n"
        # for index, out in enumerate(self.predicted_outputs.data):
        #     rep += f"PredictedOutputs[{index}]:\n" + out + "\n"
        #     rep += "=============\n"
        rep += "=============\n"
        # rep += "Score:\n" + str(self.score) + "\n"
        # rep += "=============\n"
        return rep


class Validation(BaseModel):
    """Validation"""

    outputs: ProgramOutputs


class CodingEnv(BaseModel):
    """Coding Environment"""

    # From Dataset
    dataset_frame: DatasetFrame

    # From Model
    model_predictions: List[ModelPrediction] = []

    # From Validation
    validations: List[Validation] = []

    @property
    def prompt(self):
        """Convert the code env to a prompt"""
        prompt = str(self.dataset_frame)
        prompt += "=============\n"
        for index, (model_prediction, validation) in enumerate(
            zip(self.model_predictions, self.validations)
        ):
            prompt += f"Code[{index}]:\n" + str(model_prediction) + "\n"
            prompt += "=============\n"
            prompt += f"Output[{index}]:\n" + str(validation) + "\n"
            prompt += "=============\n"
        return prompt

    @property
    def score(self):
        """Calculate the score"""
        res = 0
        model_prediction_list = self.dataset_frame.expected_outputs.data
        validation_list = self.validations[-1].outputs.data
        for model_prediction, validation in zip(
            model_prediction_list, validation_list
        ):
            res += text_similarity(
                model_prediction,
                validation,
            )
        res = res / len(self.model_predictions)
        return res


# @no_type_check
class Node(BaseModel):
    """Node for MCTS"""

    state: CodingEnv  # type: ignore

    parent: "Node" = None  # type: ignore
    children: List["Node"] = []  # type: ignore
    wins: float = 0
    visits: int = 1
    untried_actions: List[ModelPrediction] = []
