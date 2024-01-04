"""CLI interface for myugpt project.
"""

from myugpt.dataset import CodeContestsDataset
from myugpt.gpt import MyuGPT
from myugpt.mcts import mcts
from myugpt.schema import CodingEnv


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m myugpt` and `$ myugpt `.
    """
    myugpt = MyuGPT()
    dataset = CodeContestsDataset()

    frame = dataset[0]

    env = CodingEnv(
        dataset_frame=frame,
    )

    print(env.prompt)

    result = mcts(env, 10, myugpt)

    print("=" * 20)
    print("Final Result")
    print("=" * 20)
    print(result)
    print("=" * 20)
