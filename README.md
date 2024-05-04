# cvbydl2024

## Setup

Create and activate a conda environment with the necessary dependencies.

```
conda create --name cvbydl --file requirements.txt

conda activate cvbydl
```

## Run

1. Either use the [original dataset](https://arxiv.org/abs/2301.08880) and rescale it by running `python rescale_dataset.py` or directly download the [rescaled dataset]().
2. In `main.py` specify the locations TRAIN_DIR and TEST_DIR.
3. Run `python main.py`.
