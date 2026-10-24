# Homework 4: Evaluation Metrics for Classification

[Homework instructions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md) · [Module materials](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/04-evaluation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivazin/ml-zoomcamp/blob/main/hw-04/homework_04.ipynb)

## Open in Colab

Click the badge, then select **Runtime → Run all**. The notebook installs missing libraries and downloads the dataset automatically. It contains its own helper functions, so no manual uploads or other project files are needed.

The first run requires internet access. The Colab link opens the version published on GitHub's `main` branch; local changes appear there after they are pushed.

## Run locally

With [uv](https://docs.astral.sh/uv/) installed, open the notebook from the project root:

```bash
make notebook-hw-04
```

Run the cells in order or use **Run All Cells**. Stop JupyterLab with `Ctrl+C` in the terminal. The local Python version is 3.14; the project supports Python 3.12 and newer.

To run the same experiments as a script:

```bash
cd hw-04
uv run python solution.py
```

The script prints intermediate tables and computed choices. It downloads the dataset on its first run and checks the release checksum. It does not write result JSON or CSV files.

To execute the notebook without opening a browser:

```bash
cd hw-04
uv run jupyter nbconvert --to notebook --execute homework_04.ipynb --inplace
```

For an editor such as VS Code, prepare the environment and select `hw-04/.venv/bin/python` as the notebook interpreter:

```bash
cd hw-04
uv sync --locked
```

## Working notes

- This uses the same dataset as HW-03, but a different split seed. Reusing the old split would change the experiments.
- Pass probabilities to ROC AUC, not thresholded predictions. AUC measures ranking; precision, recall, and F1 describe decisions at a particular threshold.
- A precision–recall plot is useful for intuition. Use the numerical threshold table for the actual comparison, and watch for the uninformative point where both metrics are zero.
- Keep the model fixed while exploring thresholds. Otherwise, you are changing two things at once.
- Fit the encoder separately inside each cross-validation fold. Validation rows should not shape the feature representation used for training.
- Use the same folds for all regularization settings, and keep full precision until the assignment asks for rounding.
- Leave the test set aside while choosing thresholds and model settings.

## Where to work

`homework_04.ipynb` walks through each question with tables and plots. `solution.py` provides a terminal run of the calculations. The downloaded assignment is saved in `homework.md`; `uv.lock` records the local dependencies.
