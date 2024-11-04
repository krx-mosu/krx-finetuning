from openai import OpenAI

client = OpenAI(
    api_key="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"
)
import json
import pandas as pd
from datasets import load_dataset

# OpenAI API 설정


# KMMLU 데이터셋 로드
def load_kmmlu_data():
    selected_configs = ["Management", "Economics", "Accounting", "Taxation"]
    data = {}

    for config in selected_configs:
        dataset = load_dataset("HAERAE-HUB/KMMLU", config)
        # config 이름을 키로 사용하여 각각 저장
        data[config] = {
            "train": dataset["train"].to_pandas(),
            "dev": dataset["dev"].to_pandas(),
            "test": dataset["test"].to_pandas(),
        }
    print(f"Dataset loaded for configs: {selected_configs}")
    return data


# 질문 필터링 함수: GPT API를 통해 질문을 분류
def classify_question_to_field(question_text):
    classification_prompt = f"""다음 질문이 어떤 분야에 해당하는지 선택하세요:
1. Management
2. Economics
3. Accounting
4. Taxation
질문: {question_text}
해당 질문은 위의 4가지 분야 중 어느 분야에 해당합니까? 관련 분야의 이름만 반환하세요."""

    response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a classifier that assigns questions to specific fields.",
                },
                {"role": "user", "content": classification_prompt},
            ],
            max_tokens=10,
            temperature=0,
        )
        .choices[0]
        .message["content"]
        .strip()
    )

    # 결과가 4가지 중 하나인지 확인하고 필터링
    if response in ["Management", "Economics", "Accounting", "Taxation"]:
        return response
    else:
        return None  # 다른 분야일 경우 필터링


# 공통 함수: 질문과 선택지 생성
def generate_question_and_choices(prompt_text):
    question_response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that generates questions and choices.",
                },
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        .choices[0]
        .message["content"]
        .strip()
    )

    choices_prompt = f"{question_response}에 대한 선택지를 4개 생성해 주세요."
    choices_response = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that generates multiple-choice options.",
                },
                {"role": "user", "content": choices_prompt},
            ],
            max_tokens=200,
            temperature=0.5,
        )
        .choices[0]
        .message["content"]
        .strip()
    )

    return question_response, choices_response


# 공통 Step 처리 함수: Step 1(풀이 생성)과 Step 2(정답 생성)를 처리
def generate_prompt_response_pair(prompt_step_1):
    response_step_1 = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a tutor that provides explanations for answers.",
                },
                {"role": "user", "content": prompt_step_1},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        .choices[0]
        .message["content"]
        .strip()
    )

    prompt_step_2 = f"{prompt_step_1} {response_step_1}\n### 정답:"
    response_step_2 = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that gives the final answer based on explanations.",
                },
                {"role": "user", "content": prompt_step_2},
            ],
            max_tokens=50,
            temperature=0.3,
        )
        .choices[0]
        .message["content"]
        .strip()
    )

    return {"prompt": prompt_step_1, "response": response_step_1}, {
        "prompt": prompt_step_2,
        "response": response_step_2,
    }


# 문제 유형별 템플릿 정의
templates = {
    "finance_agent": """### 질문: {question}
### df.head()
|    | Symbol | Series | Date       | Prev Close | Open Price | High Price | Low Price | Last Price | Close Price | Average Price | Total Traded Quantity | Turnover | No. of Trades | Deliverable Qty | % Dly Qt to Traded Qty |
|----:|--------:|--------:|------------|------------:|------------:|------------:|-----------:|------------:|-------------:|---------------:|-----------------------:|----------:|---------------:|----------------:|-----------------------:|
|  0 | GODREJIND | EQ | 15-May-2017 | 564.6 | 581 | 584 | 568.5 | 578.9 | 578.55 | 578.09 | 797171 | 4.60836e+08 | 21649 | 360927 | 45.28 |
### 선택지:
{choices}
### 정답:""",
    "domestic_company": """### 질문: {question}
### 선택지:
{choices}
### 정답:""",
    "accounting": """### 질문: {question}
### 선택지:
{choices}
### 정답:""",
    "financial_market": """### 질문: {question}
### 선택지:
{choices}
### 정답:""",
}


# 문제 유형별 데이터 생성 함수
def generate_data_for_type(data_type, question_text):
    question, choices = generate_question_and_choices(question_text)
    prompt_step_1 = templates[data_type].format(question=question, choices=choices)
    return generate_prompt_response_pair(prompt_step_1)


# 데이터 저장 함수
def save_to_json(data, filename="kmmlu_training_data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filename}")


# KMMLU 데이터셋을 사용하여 학습 데이터 생성
def generate_kmmlu_based_data():
    kmmlu_data = load_kmmlu_data()
    data = []
    field_counts = {"Management": 0, "Economics": 0, "Accounting": 0, "Taxation": 0}

    # 각 config (Management, Economics, ...)에 대해 데이터 생성
    for config, splits in kmmlu_data.items():
        for split_name, df in splits.items():
            for question_text in df["question"].dropna():
                # 질문을 4가지 분야 중 하나로 분류
                field = classify_question_to_field(question_text)

                # 분류 결과가 4가지 중 하나인 경우에만 학습 데이터 생성
                if field:
                    field_counts[field] += 1
                    step_1, step_2 = generate_data_for_type(field, question_text)
                    data.append(step_1)
                    data.append(step_2)

    # 필터링 결과 출력
    print("Field Counts after Classification:")
    print(field_counts)

    # Step 1, 2 데이터 샘플 출력
    if data:
        print("\nSample Step 1 Data:", data[0])
        print("\nSample Step 2 Data:", data[1])

    # 데이터 부족 경고
    for field, count in field_counts.items():
        if count < 500:
            print(
                f"Warning: {field} 분야의 데이터가 {count}개로 부족합니다. 추가 데이터가 필요합니다."
            )

    save_to_json(data)


# 실행
generate_kmmlu_based_data()
