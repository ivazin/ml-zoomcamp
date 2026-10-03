# Homework 2: Linear Regression

Solutions for [ML Zoomcamp 2026 - Homework 2](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/02-regression/homework.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivazin/ml-zoomcamp/blob/main/hw-02/homework_02.ipynb)

## Running the Python Script

```bash
cd hw-02
uv run python solution.py
```

---

## Running the Jupyter Notebook (`homework_02.ipynb`)

### Option A: In VS Code / Cursor / IDE (Recommended)
1. Install `ipykernel` into the local virtual environment:
   ```bash
   cd hw-02
   uv add --dev ipykernel
   ```
2. Open `homework_02.ipynb`.
3. In the top-right corner, click **Select Kernel** -> **Python Environments...** and select `./hw-02/.venv/bin/python`.
4. Run cells with `Shift + Enter` or click **Run All**.

### Option B: In Browser via JupyterLab
1. Install `jupyterlab` as a dev dependency:
   ```bash
   cd hw-02
   uv add --dev jupyterlab
   ```
2. Start JupyterLab:
   ```bash
   uv run jupyter lab
   ```

### Option C: Run Headless (CLI execution without opening a UI)
To execute all notebook cells and save the outputs in-place:
```bash
cd hw-02
uv run --with nbconvert,ipykernel jupyter nbconvert --to notebook --execute homework_02.ipynb --inplace
```

### Option D: Open in Google Colab
Click the badge above or navigate directly to:
[Open homework_02.ipynb in Colab](https://colab.research.google.com/github/ivazin/ml-zoomcamp/blob/main/hw-02/homework_02.ipynb)

