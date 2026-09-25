# Homework 1: Intro to Machine Learning

Solutions for [ML Zoomcamp 2026 - Homework 1](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/01-intro/homework.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivazin/ml-zoomcamp/blob/main/hw-01/homework_01.ipynb)

## Running the Python Script

```bash
cd hw-01
uv run python solution.py
```

---

## Running the Jupyter Notebook (`homework_01.ipynb`)

### Option A: In VS Code / Cursor / IDE (Recommended)
1. Install `ipykernel` into the local virtual environment:
   ```bash
   cd hw-01
   uv add --dev ipykernel
   ```
2. Open `homework_01.ipynb`.
3. In the top-right corner, click **Select Kernel** -> **Python Environments...** and choose the kernel located at `./hw-01/.venv/bin/python`.
4. Run cells with `Shift + Enter` or click **Run All**.

### Option B: In Browser via JupyterLab
1. Install `jupyterlab` as a dev dependency (if not already installed):
   ```bash
   cd hw-01
   uv add --dev jupyterlab
   ```
2. Start JupyterLab:
   ```bash
   uv run jupyter lab
   ```

### Option C: Run Headless (CLI execution without opening a UI)
To execute all notebook cells and save the outputs in-place:
```bash
cd hw-01
uv run --with nbconvert,ipykernel jupyter nbconvert --to notebook --execute homework_01.ipynb --inplace
```

### Option D: Open in Google Colab
Click the badge above or navigate directly to:
[Open homework_01.ipynb in Colab](https://colab.research.google.com/github/ivazin/ml-zoomcamp/blob/main/hw-01/homework_01.ipynb)