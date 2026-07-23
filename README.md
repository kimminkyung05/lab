# Lab

해양 운동 시계열 예측과 자유 자기회귀(AR) 평가를 위한 Jupyter Notebook 프로젝트입니다.

## 데이터베이스
본 프로젝트에서는 아래 링크의 데이터를 사용합니다.
- [lrf-sswu Repository](https://github.com/yangjunahn/lrf-sswu/tree/main)


## 실행

`00_experiment_runner.ipynb`만 처음부터 실행합니다.

1. `01_data_preparation.ipynb` — 데이터 전처리
2. `02_models_and_losses.ipynb` — 모델 및 Loss 정의
3. `03_training_and_optuna.ipynb` — 학습 및 Optuna
4. `04_ar_evaluation_and_figures.ipynb` — 200-step AR 평가 및 시각화

## 설정

모든 설정은 `00_experiment_runner.ipynb`에서 관리합니다.

- 변수: `state_cols`, `target_cols`, `report_col`
- 모델: `model_names`
- Loss: `loss_type`
- 학습: `epoch`, `batch_size`, `lr`, `patience`
- Optuna: `run_optuna`, `optuna_n_trials`

기본 모델은 `BiLSTM`, `TCN`, `Transformer`이며 `db` Loss를 사용합니다.

## 평가

100-step 예측을 반복해 총 200-step 자유 AR 평가를 수행합니다.

- `RMSE@100`, `MAE@100`
- `RMSE@200`, `MAE@200`
- Test window별 RMSE 표준편차

## 출력

- 결과: `results_ar_eval_200/<experiment_name>/`
- Checkpoint: `loss_checkpoints/<experiment_name>/`

## ACF / FFT

`acf_fft.ipynb`는 Heave 신호의 ACF와 FFT를 이용해 주기 특성을 분석합니다.
