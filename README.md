
# MyuGPT

[![codecov](https://codecov.io/gh/Cardinal-Robo-Taxi/MyuGPT/branch/main/graph/badge.svg?token=MyuGPT_token_here)](https://codecov.io/gh/Cardinal-Robo-Taxi/MyuGPT)
[![CI](https://github.com/Cardinal-Robo-Taxi/MyuGPT/actions/workflows/main.yml/badge.svg)](https://github.com/Cardinal-Robo-Taxi/MyuGPT/actions/workflows/main.yml)

MyuZero Paper: https://arxiv.org/abs/1911.08265

MyuZero uses AI guided Monte Carlo tree search to make good decisions and hence play games like Atari, Go, Chess, Shogi at a super-human level.
Tesla Has Shown that it has recently applied a similar approach of AI Guided Tree Search for Path Planning. The difference being, at the moment Tesla likely uses their hard-coded simulator for training (along with their large dataset of user data).
ChatGPT is a generative AI model which takes the a programming problem statement as input along with the current code and its output and produces new code to process as output

There is potential to build a super human coding agent using ChatGPT and MyuZero

# Datasets

AlphaCode's Code Contests Dataset
- https://huggingface.co/datasets/deepmind/code_contests

CodeForces Dataset
- https://www.kaggle.com/datasets/immortal3/codeforces-dataset

LeetCode Dataset
- https://www.kaggle.com/datasets/gzipchrist/leetcode-problem-dataset
- 1,825 Leetcode problems and was last updated in April 2021


## Usage

```bash
$ python -m myugpt
#or
$ myugpt
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
