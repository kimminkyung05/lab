# Lab

## 노트북 실행 순서

아래 노트북을 **같은 Jupyter 커널**에서 순서대로 실행합니다.

1. `notebooks/00_experiment_runner.ipynb` — 설정 변경, 학습, AR 평가, 수식과 결과 그래프를 한 노트북에서 반복 실행
2. `notebooks/01_data_preparation.ipynb` — 데이터 로드, 분할, 스케일링, PCA, 시퀀스 데이터셋 생성
3. `notebooks/02_models_and_losses.ipynb` — 모델 및 전체 loss 함수 정의
4. `notebooks/03_training_and_optuna.ipynb` — 선택한 모델 학습 및 Optuna 파라미터 탐색
5. `notebooks/04_ar_evaluation_and_figures.ipynb` — AR 평가 지표와 시각화 생성

학습 전에 `01_data_preparation.ipynb`의 `config["model_names"]`와
`config["loss_type"]`를 원하는 값으로 변경합니다.

빠르게 여러 조건을 비교하려면 `00_experiment_runner.ipynb`만 사용하면 됩니다.
첫 번째 설정 셀의 `models`, `loss`, `epochs` 등을 수정한 뒤, 다음 실행 셀을 다시 실행하세요.
