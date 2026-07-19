"""Create focused notebooks from the monolithic experiment notebook.

Run from the repository root: ``python tools/split_notebooks.py``.
The source notebook remains unchanged; generated notebooks deliberately contain
fresh (empty) outputs so they are safe to commit and rerun.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lt_db.ipynb"
OUTPUT_DIR = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def select(cells: list[dict], indexes: list[int]) -> list[dict]:
    selected = []
    for index in indexes:
        cell = cells[index].copy()
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        selected.append(cell)
    return selected


def write_notebook(filename: str, title: str, description: str, cells: list[dict]) -> None:
    notebook = {
        "cells": [markdown(f"# {title}\n\n{description}\n")] + cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUTPUT_DIR / filename).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def experiment_runner_cells() -> list[dict]:
    return [
        markdown(
            "## 실행 설정\n\n"
            "아래 값만 수정한 뒤 다음 셀을 실행하세요. 같은 커널에서 이 셀과 실행 셀을 반복 실행할 수 있습니다.\n"
        ),
        code(
            '''RUN_CONFIG = {
    # 실행할 모델과 loss
    "models": ["LSTM", "Transformer"],
    "loss": "db",  # mse, ep, dilate, tildeq, db, kmbdf, cp

    # 결과 저장 폴더 이름
    "experiment": "lstm_transformer_db",

    # None이면 원본 기본 config 값을 사용합니다.
    "epochs": 1000,
    "batch_size": None,
    "lr": None,
    "patience": None,
    "input_len": None,
    "output_len": None,

    # 빠른 비교는 False, 파라미터 탐색까지 하려면 True
    "run_optuna": False,
    "optuna_trials": 30,

    # 결과 그래프에 표시할 test sample 번호
    "sample_idx": 0,
}
'''
        ),
        markdown("## 학습 · AR 평가 · 수식 · 결과 그래프\n\n이 셀 하나가 학습부터 200-step AR 평가와 시각화까지 실행합니다.\n"),
        code(
            '''from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Math, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / "lt_db.ipynb").exists():
    ROOT = ROOT.parent
NOTEBOOK_PATH = ROOT / "lt_db.ipynb"

with NOTEBOOK_PATH.open(encoding="utf-8") as file:
    source_cells = json.load(file)["cells"]

# 원본 구현을 불러온 후, 위 RUN_CONFIG로 기본 설정을 덮어씁니다.
namespace = {"__name__": "__main__"}
for cell_index in (0, 2):
    exec("".join(source_cells[cell_index]["source"]), namespace)

config = namespace["config"]
config["model_names"] = list(RUN_CONFIG["models"])
config["loss_type"] = RUN_CONFIG["loss"]
config["experiment_name"] = RUN_CONFIG["experiment"]
config["use_optuna_best_params"] = bool(RUN_CONFIG["run_optuna"])
config["run_optuna"] = bool(RUN_CONFIG["run_optuna"])

for run_key, config_key in {
    "epochs": "epoch",
    "batch_size": "batch_size",
    "lr": "lr",
    "patience": "patience",
    "input_len": "input_len",
    "output_len": "model_output_len",
    "optuna_trials": "optuna_n_trials",
}.items():
    if RUN_CONFIG[run_key] is not None:
        config[config_key] = RUN_CONFIG[run_key]

if RUN_CONFIG["output_len"] is not None:
    config["eval_output_len"] = int(RUN_CONFIG["output_len"]) * 2
    config["metric_output_len"] = config["eval_output_len"]

config["target_dim"] = len(config["target_cols"])
config["heave_idx"] = config["target_cols"].index(config["report_col"])
config["experiment_dir"] = os.path.join(config["results_dir"], config["experiment_name"])
config["optuna_best_params_path"] = os.path.join(
    config["experiment_dir"], "optuna_best_params_by_model.json"
)
config["checkpoint_dir"] = os.path.join(config["checkpoint_root"], config["experiment_name"])
os.makedirs(config["experiment_dir"], exist_ok=True)
os.makedirs(config["checkpoint_dir"], exist_ok=True)

print("=" * 70)
print("실험 설정")
print(f"모델: {config['model_names']} | loss: {config['loss_type']}")
print(f"PCA: 누적 설명분산 {config['pca_n_components']:.0%} 이상")
print("=" * 70)

# 데이터 준비 → 모델/loss 정의 → 학습 → AR 평가
PIPELINE_CELLS = (4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18)
for cell_index in PIPELINE_CELLS:
    source = "".join(source_cells[cell_index]["source"])
    source = source.replace(
        "display(detailed_metrics_df)",
        "print(detailed_metrics_df.to_string(index=False))",
    )
    exec(source, namespace)

# 선택한 loss의 수식을 출력합니다.
loss_formulas = {
    "mse": r"\\mathrm{MSE}=\\frac{1}{N}\\sum_{i=1}^{N}(\\hat{y}_i-y_i)^2",
    "ep": r"\\mathcal{L}_{EP}=\\mathrm{MSE}+\\lambda_{peak}\\mathcal{L}_{peak}",
    "dilate": r"\\mathcal{L}_{DILATE}=\\alpha\\mathcal{L}_{shape}+(1-\\alpha)\\mathcal{L}_{temporal}",
    "tildeq": r"\\mathcal{L}_{\\tilde{Q}}=\\mathrm{MSE}+\\lambda_a\\mathcal{L}_{amp}+\\lambda_p\\mathcal{L}_{phase}+\\lambda_c\\mathcal{L}_{corr}",
    "db": r"\\mathcal{L}_{DB}=\\mathcal{L}_{base}+\\mathcal{L}_{distribution}+\\mathcal{L}_{boundary}",
    "kmbdf": r"\\mathcal{L}_{KMDBF}=\\mathcal{L}_{base}+\\mathcal{L}_{kernel}+\\mathcal{L}_{frequency}",
    "cp": r"\\mathcal{L}_{CP}=\\mathcal{L}_{base}+\\mathcal{L}_{perceptual}",
}
display(Markdown(f"### 현재 loss: `{config['loss_type']}`"))
display(Math(loss_formulas[config["loss_type"]]))

# 지정한 sample의 입력·정답·모델별 AR 예측을 한 그래프에 표시합니다.
results = namespace["results"]
ar_results = namespace["ar_results"]
test_dataset = namespace["test_dataset"]
heave_idx = namespace["heave_idx"]
INPUT_LEN = namespace["INPUT_LEN"]
EVAL_OUTPUT_LEN = namespace["EVAL_OUTPUT_LEN"]

sample_idx = int(RUN_CONFIG["sample_idx"])
if not 0 <= sample_idx < len(test_dataset):
    raise IndexError(f"sample_idx는 0부터 {len(test_dataset) - 1} 사이여야 합니다.")

xb, _ = test_dataset[sample_idx]
x_pre_scaled = namespace["model_to_pre_pca_scaled"](xb.cpu().numpy())
x_input = namespace["inverse_scale_feature"](
    x_pre_scaled[:, namespace["pre_pca_input_cols"].index(config["report_col"])],
    config["report_col"],
).reshape(-1)

first_model = next(model for model in config["model_names"] if model in ar_results)
y_true = ar_results[first_model]["trues_orig"][sample_idx, :EVAL_OUTPUT_LEN]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.figure(figsize=(12, 5))
plt.plot(np.arange(-INPUT_LEN, 0), x_input, "--", color="gray", label="Input heave")
plt.plot(np.arange(EVAL_OUTPUT_LEN), y_true, color="black", linewidth=2, label="Ground truth")
for index, model_name in enumerate(config["model_names"]):
    if model_name not in ar_results:
        continue
    prediction = ar_results[model_name]["preds_orig"][sample_idx, :EVAL_OUTPUT_LEN]
    plt.plot(np.arange(EVAL_OUTPUT_LEN), prediction, color=colors[index % len(colors)], label=f"{model_name} prediction")

plt.axvline(0, color="black", linestyle=":", alpha=0.6)
plt.title(f"{config['experiment_name']} | sample {sample_idx} | 200-step free AR")
plt.xlabel("Time step")
plt.ylabel(config["report_col"])
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.show()
'''
        ),
    ]


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = source["cells"]
    OUTPUT_DIR.mkdir(exist_ok=True)

    write_notebook(
        "00_experiment_runner.ipynb",
        "실험 실행 노트북",
        "모델, loss, 학습 조건을 수정하고 같은 노트북에서 반복 실행합니다.",
        experiment_runner_cells(),
    )
    write_notebook(
        "01_data_preparation.ipynb",
        "01. Data preparation",
        "Load `merged_data.csv`, validate the time series, split by source file, and build scaled/PCA sequence datasets.",
        select(cells, [0, 2, 3, 4, 5, 6, 7]),
    )
    write_notebook(
        "02_models_and_losses.ipynb",
        "02. Models and loss functions",
        "Model and loss-function library. Run this after data preparation in the same kernel, or use `run.py` for a full experiment.",
        select(cells, [0, 2, 4, 8, 9, 10, 11, 12]),
    )
    write_notebook(
        "03_training_and_optuna.ipynb",
        "03. Training and Optuna",
        "Run after notebooks 01 and 02 in the same kernel. Select `config['model_names']` and `config['loss_type']` before running the final cell.",
        select(cells, [13, 14, 15, 16]),
    )
    write_notebook(
        "04_ar_evaluation_and_figures.ipynb",
        "04. AR evaluation and figures",
        "Run after notebook 03 in the same kernel. Produces 200-step autoregressive metrics and the sample/paper figures.",
        select(cells, [17, 18, 19, 20, 21, 22, 23, 24, 25]),
    )


if __name__ == "__main__":
    main()
