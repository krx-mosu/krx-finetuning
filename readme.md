````markdown
# KMMLU 금융시장 데이터셋 처리 및 확장 프로젝트

## 프로젝트 개요

이 프로젝트는 KMMLU(Korean Multiple-choice Multi-task Language Understanding) 데이터셋을 기반으로 금융시장 관련 문제들을 고품질의 학습 데이터셋으로 변환하고 확장하는 것을 목적으로 합니다. 최종적으로 이 데이터셋은 LLM(Large Language Model)의 금융시장 관련 문제 해결 능력을 향상시키기 위한 파인튜닝에 활용됩니다.

## 목표

1. 기존 KMMLU 경제 문제들을 금융시장 특화 문제로 변환
2. 고품질의 추가 문제 생성을 통한 데이터셋 확장
3. 각 문제에 대한 상세한 풀이 과정과 정답 생성
4. 정확히 2000개의 프롬프트-응답 쌍(1000개 문제 × 2단계) 구축

## 데이터 처리 파이프라인

### 1. 데이터 수집

- KMMLU Economics 데이터셋 다운로드
  - train, dev, test 데이터셋 활용
  - 출처: https://huggingface.co/datasets/HAERAE-HUB/KMMLU

### 2. 데이터 필터링

- 금융시장 관련 문제 선별
  - GPT-4를 활용한 관련성 판단
  - 금융 시장, 투자, 거래 관련 개념 포함 여부 확인
  - 실제 금융 시나리오 기반 문제 선별

### 3. 문제 변환 및 확장

- 기존 문제 변환

  - 4개 선택지를 7개로 확장
  - 금융시장 맥락에 맞게 내용 수정
  - 명확한 단일 정답 보장

- 추가 문제 생성
  - 금융시장 관련 새로운 문제 생성
  - 다양한 주제 영역 포괄
    - 주식 시장 제도
    - 파생상품 거래
    - 금융상품 분석
    - 리스크 관리
    - 투자 전략
    - 금융 규제
    - 채권 시장
    - 외환 시장

### 4. 학습 데이터 생성

각 문제당 2단계의 데이터 생성:

1. **스텝 1: 문제 풀이**

   - Input: 문제 + "### 정답:"
   - Output: 체계적인 풀이 과정
     - 문제 분석
     - 각 선택지 검토
     - 정답 도출 과정
     - 금융 이론/실무 연계 설명

2. **스텝 2: 최종 답변**
   - Input: 문제 + "### 정답:" + 풀이 과정
   - Output: 최종 정답(알파벳)

### 5. 품질 관리

- 문제 품질 기준

  - 단일 정답 보장
  - 명확한 선택지 구분
  - 실제 금융 시장 상황 반영
  - 전문성과 실용성 균형

- 풀이 품질 기준
  - 논리적 단계별 접근
  - 금융 이론/실무 근거 제시
  - 각 선택지 분석 포함
  - 명확한 정답 도출 과정

## 기술 스택

- Python 3.8+
- OpenAI GPT-4 API
- 주요 라이브러리:
  - pandas: 데이터 처리
  - openai: GPT-4 API 통신
  - concurrent.futures: 병렬 처리
  - backoff: API 재시도 로직
  - tqdm: 진행률 표시
  - logging: 로그 관리

## 처리 성능 최적화

- 병렬 처리 적용
  - ThreadPoolExecutor 활용
  - 최대 100개 동시 처리
- API 호출 최적화
  - 재시도 로직 구현
  - 에러 처리 및 복구

## 결과물

1. **데이터 파일**
   - final*training_data*{timestamp}.json
     - 2000개의 프롬프트-응답 쌍
     - 문제-풀이-답변 구조
   - processing*stats*{timestamp}.json
     - 처리 통계 정보
2. **로그 파일**
   - processing\_{timestamp}.log
     - 상세 처리 과정
     - 에러 및 경고 기록

## 데이터 형식

```json
[
  {
    "prompt": "다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시요.\n### 질문: [문제]\n### 선택지: A~G\n### 정답:",
    "response": "[상세 풀이 과정]"
  },
  {
    "prompt": "다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시요.\n### 질문: [문제]\n### 선택지: A~G\n### 정답:\n[상세 풀이 과정]",
    "response": "[최종 답변(알파벳)]"
  }
]
```
````

## 사용 방법

1. 환경 설정

   ```bash
   pip install -r requirements.txt
   ```

2. OpenAI API 키 설정

   ```python
   OPENAI_API_KEY = "your-api-key"
   ```

3. 실행
   ```python
   python main.py
   ```

## 주의사항

1. API 비용 고려

   - GPT-4 API 호출 비용 발생
   - 대량 처리 시 비용 계산 필요

2. 실행 시간

   - 전체 처리에 상당한 시간 소요
   - 중간 결과 자동 저장 활용

3. 에러 처리
   - API 오류 자동 재시도
   - 중간 결과 보존

## 향후 개선 방향

1. 데이터 다양성 확대

   - 더 다양한 금융 시장 시나리오 포함
   - 국내외 시장 상황 반영

2. 품질 개선

   - 문제 난이도 조절 기능
   - 풀이 품질 자동 평가

3. 성능 최적화
   - 처리 속도 개선
   - 비용 효율성 향상

```

```