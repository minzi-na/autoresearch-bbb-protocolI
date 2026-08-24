# Phase 2 Process Notes

## 목적

이 문서는 BBB autoresearch의 **2단계(HPO 단계)** 를 어떻게 운영할지에 대한 기록이다.

1단계는 아키텍처 최적화 단계이고,
2단계는 1단계에서 고른 **best architecture를 고정한 상태에서 hyperparameter optimization(HPO)** 을 수행하는 단계다.

## 핵심 결론

- 1단계는 `bbb_train.py`를 수정하며 아키텍처를 탐색한다.
- 2단계는 `bbb_train.py`의 best architecture를 고정하고, `bbb_optuna.py`를 수정하며 HPO를 탐색한다.
- 2단계에서도 autoresearch를 사용할 수 있다.
- 다만 2단계에서 autoresearch의 역할은 **architecture modification engineer** 가 아니라
  **Optuna search-space design & modification engineer** 이다.

## 1단계와 2단계의 차이

### 1단계

- 목적: 더 좋은 모델 구조 찾기
- 수정 대상: `bbb_train.py`
- 고정 대상:
  - `bbb_prepare.py`
  - `BASE_CONFIG`의 고정 하이퍼파라미터
- 실험 단위:
  - 아키텍처 변경 1개
  - `python bbb_train.py` 실행
  - 10-seed 결과 비교 후 keep/discard

### 2단계

- 목적: 고정된 best architecture에 대해 더 좋은 hyperparameter setting 찾기
- 수정 대상: `bbb_optuna.py`
- 고정 대상:
  - `bbb_prepare.py`
  - `bbb_train.py`의 best architecture
- 실험 단위:
  - Optuna search space / objective / sampler / pruner 설계 변경
  - `python bbb_optuna.py` 실행
  - best trial 성능과 탐색 효율 비교 후 keep/discard

## 2단계에서 autoresearch가 바꾸는 것

2단계에서 autoresearch는 더 이상 다음을 제안하지 않는다.

- pooling 구조 변경
- SGU 변경
- residual 구조 변경
- projection 구조 변경

대신 다음을 제안하고 수정한다.

- `d_model` 탐색 후보 범위
- `d_ffn` 탐색 후보 범위
- `dropout` 범위
- `lr` / `weight_decay` 범위
- `batch_size` 후보
- Optuna sampler 설정
- Optuna pruner 설정
- objective 정의
- trial 수 / timeout / staged evaluation 전략
- top-k reevaluation 방식

즉, 2단계에서 바뀌는 것은 **모델 구조**가 아니라 **탐색기 설계**다.

## Step 2 문구 설계

2단계용 외부 루프는 아래처럼 정의한다.

### Step 1. Best architecture 고정

- 1단계에서 선정된 best architecture를 `bbb_train.py`에 고정한다.
- 모델 구조는 더 이상 변경하지 않는다.
- HPO 대상 파라미터만 탐색 가능하도록 준비한다.

### Step 2. Agent가 Optuna 탐색 설계 변경 제안

- `bbb_optuna.py`의 search space와 탐색 전략을 수정한다.
- 예:
  - search space 범위 조정
  - sampler/pruner 변경
  - objective 정의 수정
  - trial budget 조정
  - seed 평가 전략 변경
- 아키텍처 변경 제안은 하지 않는다.

### Step 3. `bbb_optuna.py` 수정 후 실행

- Optuna study를 실행한다.
- 각 trial은 고정된 best architecture를 학습 및 평가한다.
- 여러 trial 중 best trial을 선택한다.
- 필요하면 상위 trial에 대해 추가 재평가를 수행한다.

### Step 4. 결과 파싱

- 주 판단 지표:
  - best validation/internal scaffold ROC-AUC
- 보조 지표:
  - `roc_s`: internal scaffold test ROC-AUC의 10-seed mean
  - `roc_ext`: external 예측확률의 10-seed soft voting ensemble ROC-AUC
  - `roc_holdout`: holdout 예측확률의 10-seed soft voting ensemble ROC-AUC
  - 탐색 효율
  - 안정성
- 이번 탐색 설계 변경이 실제로 더 좋은 best trial을 찾았는지 평가한다.

## 2단계 Step 3의 의미

중요한 점은, 2단계에서는 Step 3가 더 이상

- "`bbb_train.py`를 수정하고 한 번 실행"

의 의미가 아니라는 점이다.

2단계 Step 3의 본질은:

- "`bbb_optuna.py`를 수정하고 실행"
- "Optuna가 여러 trial을 돌리면서 `bbb_train.py` backend를 반복 호출"

이다.

즉 `bbb_train.py`는 2단계에서 직접적인 실험 엔트리포인트라기보다,
**best architecture가 고정된 학습 backend** 역할을 하게 된다.

## 필요한 파일

2단계에서 필요한 핵심 파일은 다음과 같다.

- `bbb_prepare.py`
- `bbb_train.py`
- `bbb_optuna.py`
- `bbb_optuna_program.md`

### 각 파일의 역할

#### `bbb_prepare.py`

- 데이터 로딩
- 피처 생성
- scaffold split
- RDKit normalization
- 평가 함수 제공

2단계에서도 **필요하다.**
다만 수정 대상은 아니고 **고정 의존 파일**이다.

#### `bbb_train.py`

- 1단계에서 고른 best architecture 보관
- 모델 생성/학습/평가 backend

2단계에서는 아키텍처를 수정하지 않는다.

#### `bbb_optuna.py`

- Optuna study 실행 파일
- search space 정의
- sampler / pruner / objective 정의
- trial 결과 로깅
- 최종 reevaluation에서 `roc_s`는 internal scaffold test 10-seed mean으로 계산
- 최종 reevaluation에서 `roc_ext` / `roc_holdout`은 10-seed soft voting ensemble로 계산

2단계의 주 수정 대상이다.

#### `bbb_optuna_program.md`

- 2단계용 operating 문서
- autoresearch agent가 무엇을 바꾸고 무엇을 바꾸지 말아야 하는지 정의

## 왜 `bbb_prepare.py`가 여전히 필요한가

질문이 있었던 부분은 다음이다.

- "`bbb_prepare.py`는 baseline parameter가 고정인데 2단계에서도 필요한가?"

답은 **필요하다** 이다.

이유:

- `bbb_prepare.py`는 주로 데이터/평가 인프라를 담당한다.
- 2단계에서 HPO로 열고 싶은 파라미터는 주로 `bbb_train.py` 쪽 학습 설정이다.
- 따라서 2단계에서 풀어야 하는 것은 `bbb_prepare.py`가 아니라
  `bbb_train.py`의 고정 config 의존성이다.

즉:

- `bbb_prepare.py`는 그대로 둔다.
- `bbb_train.py`를 HPO 가능하게 리팩터링한다.

## 왜 1단계 완료 후 `bbb_train.py`를 리팩터링해야 하는가

1단계가 끝난 뒤에는 `bbb_train.py`를 한 번 리팩터링하는 것이 맞다.

이 리팩터링의 목적은:

- 아키텍처를 바꾸는 것

이 아니라

- **best architecture를 유지한 채 Optuna가 trial별 config를 주입할 수 있게 만드는 것**

이다.

현재 1단계용 `bbb_train.py`는 다음 특징이 있다.

- `BASE_CONFIG` 중심의 고정 실행
- 바로 10-seed 평가까지 수행하는 스크립트형 구조
- 외부에서 `d_model`, `lr`, `batch_size` 등을 trial 단위로 주입하기 어려움

그래서 2단계 전환 시에는 보통 아래처럼 정리해야 한다.

- `build_model(mod_dims, config)`
- `make_optimizer(model, config)`
- `run_one_seed(..., config, seed)`
- `run_k_seeds(..., config, seeds)`

이렇게 하면 `bbb_optuna.py`가 trial마다 config를 생성해서 `bbb_train.py` backend에 전달할 수 있다.

## 최종 운영 원칙

- 1단계가 끝나기 전에는 HPO용 리팩터링을 섞지 않는다.
- 1단계 종료 후 best architecture를 확정한다.
- 그 다음 `bbb_train.py`를 HPO-friendly backend로 리팩터링한다.
- `bbb_optuna.py`를 별도 엔트리포인트로 두고 2단계를 운영한다.
- 2단계의 autoresearch는 아키텍처 변경이 아니라 Optuna 탐색 설계 최적화에 집중한다.
- Phase 2 최종 보고 시 `roc_s`는 internal scaffold test 10-seed mean, `roc_ext` / `roc_holdout`은 10-seed soft voting ensemble로 통일한다.

## 요약

- 1단계: `bbb_train.py`를 수정하며 아키텍처 탐색
- 2단계: `bbb_train.py`는 고정, `bbb_optuna.py`를 수정하며 HPO 탐색
- `bbb_prepare.py`는 2단계에서도 필요하지만 수정 대상은 아님
- 2단계 시작 전 `bbb_train.py`는 HPO를 위해 리팩터링하는 것이 바람직함
- 2단계 최종 지표 정의는 `roc_s` = internal scaffold test 10-seed mean, `roc_ext` / `roc_holdout` = 10-seed soft voting ensemble이다.
