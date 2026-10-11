# Homework 3: Classification

[Homework instructions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/03-classification/homework.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivazin/ml-zoomcamp/blob/main/hw-03/homework_03.ipynb)

## Open in Colab

Click the badge above, then select **Runtime → Run all**. The notebook installs missing dependencies and downloads the dataset automatically. No manual uploads are needed; the first run requires internet access.

The link opens the notebook published on GitHub's `main` branch. Local changes appear there after they are pushed.

## Run locally

You need [uv](https://docs.astral.sh/uv/) installed. From the project root, open the notebook in JupyterLab:

```bash
make notebook-hw-03
```

The command prepares the environment and opens `homework_03.ipynb`. Run the cells in order, or use **Run All Cells**. Stop the server with `Ctrl+C` in the terminal.

To run the Python script instead:

```bash
cd hw-03
uv run python solution.py
```

To execute the notebook and save its outputs without opening a browser:

```bash
cd hw-03
uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute homework_03.ipynb --inplace
```

For an editor such as VS Code, prepare the environment and select `hw-03/.venv/bin/python` as the notebook interpreter:

```bash
cd hw-03
uv sync --locked
uv pip install ipykernel
```

## Working notes

This homework connects exploratory analysis with a logistic regression model for lead conversion.

- Fit the categorical encoder on the training set, then reuse it for validation. The validation set should not decide which categories the model learns.
- Correlation and mutual information are useful clues, but neither tells the whole story about a feature's contribution to a model.
- When removing a feature, remove all of its one-hot columns and retrain. Keep the accuracy difference signed: removing a feature can improve the score.
- A smaller `C` means stronger regularization. Feature scale affects how that penalty behaves, so keep the original scale when reproducing this assignment.
- Use validation for comparisons and leave the test set aside. Round only at the steps specified in the homework.
