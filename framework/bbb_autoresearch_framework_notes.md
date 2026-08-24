# BBB AutoResearch Framework 설명

## 한 문장 요약

AutoResearch는 모델 weight를 직접 "스스로 고치는" 시스템이라기보다, **AI agent가 아키텍처를 바꾸고 실험 결과를 비교하면서 더 좋은 구조만 남기는 자동 연구 루프**입니다.

## 핵심 구조

이 프레임워크는 두 개의 루프로 이해하면 됩니다.

1. **외부 루프: 아키텍처 탐색**
   - agent가 `bbb_train.py`를 수정합니다.
   - 수정 대상은 projection, gMLP block, SGU, residual, pooling 같은 **구조 요소**입니다.
   - 실행 후 `roc_auc_scaffold`가 개선되면 **keep**, 아니면 **discard**합니다.
   - discard된 실험은 git 되돌리기로 제거되고, keep된 실험만 다음 탐색의 출발점이 됩니다.

2. **내부 루프: 일반적인 모델 학습**
   - 고정된 아키텍처 안에서 loss를 줄이도록 weight가 업데이트됩니다.
   - forward, backward, optimizer step, early stopping이 여기서 일어납니다.
   - 즉, 모델이 "스스로 학습"하는 것은 이 내부 루프이고, "어떤 구조가 더 잘 학습되는가"를 고르는 것은 외부 루프입니다.

## Keep / Discard가 의미하는 것

이 프레임워크의 핵심은 **선택 압력(selection pressure)** 입니다.

- **Keep**
  - `roc_auc_scaffold`가 이전 최고 성능보다 높음
  - 가능하면 external, holdout도 같이 좋아짐
  - 복잡도 증가가 과하지 않음

- **Discard**
  - scaffold ROC-AUC가 개선되지 않음
  - internal score만 오르고 generalization이 나빠짐
  - 코드가 복잡해졌는데 이득이 작음
  - crash 또는 학습 불안정 발생

즉, 이 시스템은 모든 아이디어를 누적하지 않고, **성능과 단순성 기준을 통과한 구조만 살아남게** 설계되어 있습니다.

## 내 모델에서 아키텍처 최적화가 일어나는 위치

BBB 모델 기준으로 최적화 포인트는 아래 네 군데입니다.

1. **Per-modality projection**
   - 서로 분포가 다른 fingerprint / descriptor / embedding을 같은 `d_model` 공간으로 보냅니다.
   - 여기서 norm, scaling, residual projection을 넣으면 modality 간 scale mismatch를 줄일 수 있습니다.

2. **gMLP backbone**
   - modality token들 사이의 상호작용을 학습하는 중심부입니다.
   - SGU 구조, gating 방식, normalization 위치, FFN 변형이 성능 차이를 크게 만듭니다.

3. **Pooling**
   - 여러 modality token을 최종 예측용 하나의 벡터로 압축합니다.
   - 단순 mean보다 gated pool, attention형 pool이 중요한 token에 더 큰 가중치를 줄 수 있습니다.

4. **Residual / training stability**
   - 같은 하이퍼파라미터에서도 residual scaling, norm ordering, clipping 같은 설계가 학습 안정성과 일반화를 바꿉니다.

## 발표에서 이렇게 설명하면 됩니다

“우리 모델은 한 번 학습되고 끝나는 구조가 아니라, AutoResearch 루프를 통해 아키텍처 후보들을 계속 생성하고 평가합니다.  
각 후보는 동일한 학습 조건에서 10-seed scaffold 평가를 받고, 성능이 개선된 구조만 keep됩니다.  
따라서 최종 모델은 사람이 수동으로 고른 단일 설계가 아니라, 반복적인 keep/discard 선택을 통과한 구조적 최적화의 결과입니다.”

## 이 그림에서 강조할 포인트

- **모델 최적화는 weight 최적화 + 구조 최적화의 2단계다**
- **discard는 실패가 아니라 탐색 공간을 줄이는 학습 신호다**
- **keep된 구조만 누적되므로, branch 자체가 점점 더 강한 아키텍처로 진화한다**
- **최종적으로는 같은 데이터, 같은 하이퍼파라미터 조건에서 더 잘 일반화되는 구조를 찾는 과정이다**
