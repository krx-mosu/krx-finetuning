import pandas as pd
import json
import time
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
import backoff
import requests
from pathlib import Path
from google.colab import files

# OpenAI API 키 설정
client = OpenAI(api_key="")
# URL 설정
KMMLU_URLS = {
    "train": "https://huggingface.co/datasets/HAERAE-HUB/KMMLU/resolve/main/data/Economics-train.csv",
    "dev": "https://huggingface.co/datasets/HAERAE-HUB/KMMLU/resolve/main/data/Economics-dev.csv",
    "test": "https://huggingface.co/datasets/HAERAE-HUB/KMMLU/resolve/main/data/Economics-test.csv",
}


class DatasetProcessor:
    def __init__(self, api_key: str, base_dir: str = "/content/kmmlu_processing"):
        self.client = OpenAI(api_key=api_key)
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self):
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.log_dir = self.base_dir / "logs"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for dir_path in [self.data_dir, self.output_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    self.log_dir / f"processing_{self.timestamp}.log", encoding="utf-8"
                ),
                logging.StreamHandler(),
            ],
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def call_openai_api(self, messages: List[Dict], temp: float = 0.7) -> str:
        return (
            self.client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=temp
            )
            .choices[0]
            .message.content
        )

    def download_dataset(self, url: str, output_path: str) -> str:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            output_path = Path(output_path)
            with open(output_path, "wb") as file, tqdm(
                desc=f"Downloading {output_path.name}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            return str(output_path)
        except Exception as e:
            logging.error(f"다운로드 실패 - URL: {url}, Error: {str(e)}")
            raise

    def is_financial_market_question(self, question: str) -> bool:
        system_prompt = """다음 질문이 금융시장 관련 문제인지 판단해주세요.
다음 조건을 만족해야 합니다:
- 금융 시장, 투자, 거래 관련 개념
- 시장 동향과 분석
- 투자 전략과 의사결정
- 실제 금융 시나리오 기반 문제해결
답변은 True 또는 False로만 해주세요."""

        try:
            response = self.call_openai_api(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temp=0.1,
            )
            return response.strip().lower() == "true"
        except Exception:
            return False

    def transform_question(
        self, question: str, choices: Dict[str, str]
    ) -> Optional[str]:
        system_prompt = """금융시장 관련 문제를 아래 형식으로 변환해주세요:

다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시요.
### 질문: [문제]
### 선택지:
A. [A보기]
B. [B보기]
C. [C보기]
D. [D보기]
E. [E보기]
F. [F보기]
G. [G보기]

주의사항:
1. 반드시 정답이 하나만 존재하도록 문제를 설계하세요
2. 다른 보기들은 명확히 오답이어야 하며, 부분적으로 맞는 내용이 없어야 합니다
3. 실제 금융 시장 상황과 시나리오를 바탕으로 작성하세요
4. 각 보기는 서로 독립적이고 중복되지 않아야 합니다
5. 정답과 오답의 길이나 형식이 유사해야 합니다
6. 모든 보기는 문제와 직접적으로 관련이 있어야 합니다"""

        try:
            return self.call_openai_api(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"원본 문제: {question}\n원본 선택지: {choices}",
                    },
                ]
            )
        except Exception as e:
            logging.error(f"문제 변환 실패: {str(e)}")
            return None

    def create_problem_solution(self, question: str) -> Optional[Tuple[Dict, Dict]]:
        try:
            # 스텝 1: 풀이 생성
            solution_prompt = question + "\n### 정답:"
            system_prompt_solution = """주어진 금융시장 관련 문제에 대해 다음 단계에 따라 체계적으로 풀이를 제시해주세요:

1. 문제 분석
   - 문제에서 요구하는 것이 무엇인지 명확히 파악
   - 주요 개념과 키워드 식별

2. 각 선택지 분석
   - 모든 선택지를 하나씩 자세히 검토
   - 각 선택지의 진위 여부를 금융 이론/실무와 연계하여 설명
   - 오답인 경우, 왜 오답인지 구체적으로 설명

3. 정답 도출 과정
   - 정답 선택의 근거를 금융 이론/법규/실무 관점에서 설명
   - 관련된 금융 개념이나 원칙 제시
   - 실제 금융 시장에서의 적용 사례 언급 (가능한 경우)

4. 최종 정답 선택
   - 위의 분석을 종합하여 최종 정답 제시
   - 다른 선택지와 비교하여 왜 이 답이 가장 적절한지 설명

풀이는 논리적이고 단계적으로 전개하되, 금융 전문 지식을 바탕으로 설명해주세요."""

            solution = self.call_openai_api(
                [
                    {"role": "system", "content": system_prompt_solution},
                    {"role": "user", "content": solution_prompt},
                ]
            )

            # 스텝 2: 최종 답변 생성
            answer_prompt = solution_prompt + "\n" + solution
            final_answer = self.call_openai_api(
                [
                    {
                        "role": "system",
                        "content": "풀이를 바탕으로 최종 정답을 선택지 중에서 선택하여 알파벳으로만 답해주세요.",
                    },
                    {"role": "user", "content": answer_prompt},
                ],
                temp=0.1,
            )

            return (
                {"prompt": solution_prompt, "response": solution},
                {"prompt": answer_prompt, "response": final_answer},
            )
        except Exception as e:
            logging.error(f"문제 풀이 생성 실패: {str(e)}")
            return None

    def generate_single_question(self) -> Optional[List[Dict]]:
        """단일 추가 문제 생성"""
        try:
            system_prompt = """다음 형식으로 금융시장 관련 문제를 생성해주세요:

        다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시요.
        ### 질문: [문제]
        ### 선택지:
        A. [A보기]
        B. [B보기]
        C. [C보기]
        D. [D보기]
        E. [E보기]
        F. [F보기]
        G. [G보기]

        주의사항:
        1. 반드시 정답이 하나만 존재하도록 문제를 설계하세요
        2. 다른 보기들은 명확히 오답이어야 하며, 부분적으로 맞는 내용이 없어야 합니다
        3. 실제 금융 시장 상황과 시나리오를 바탕으로 작성하세요
        4. 각 보기는 서로 독립적이고 중복되지 않아야 합니다
        5. 정답과 오답의 길이나 형식이 유사해야 합니다
        6. 모든 보기는 문제와 직접적으로 관련이 있어야 합니다

        문제 유형 예시:
        1. "다음 중 옳은 것을 고르시오"
        2. "다음 중 옳지 않은 것을 고르시오"
        3. "다음 설명으로 가장 적절한 것을 고르시오"

        생성할 문제 주제 예시:
        1. 주식 시장 제도 및 규정
        2. 파생상품 거래 전략
        3. 금융상품 평가 및 분석
        4. 리스크 관리 및 투자 의사결정
        5. 시장 분석 및 투자 전략
        6. 금융규제 및 법규
        7. 채권 시장 및 금리
        8. 외환 시장 및 국제 금융"""

            question = self.call_openai_api(
                [{"role": "system", "content": system_prompt}], temp=0.9
            )

            result = self.create_problem_solution(question)
            if result:
                return list(result)
            return None

        except Exception as e:
            logging.error(f"추가 문제 생성 실패: {str(e)}")
            return None

    def save_results(self, pairs: List[Dict], is_final: bool = False):
        """결과 저장"""
        file_prefix = "final" if is_final else "interim"
        output_file = (
            self.output_dir / f"{file_prefix}_training_data_{self.timestamp}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        logging.info(f"{file_prefix.capitalize()} 결과 저장 완료: {len(pairs)}개 쌍")

    def process_dataset(self, input_file: str, max_workers: int = 10) -> None:
        """데이터셋 처리 메인 파이프라인"""
        try:
            df = pd.read_csv(input_file)
            logging.info(f"데이터 로드 완료 - 총 {len(df)}개 문제")

            training_pairs = []
            save_interval = 50  # 50개 문제마다 중간 결과 저장

            # 기존 문제 처리
            filtered_questions = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_question = {
                    executor.submit(
                        self.is_financial_market_question, row["question"]
                    ): (idx, row)
                    for idx, row in df.iterrows()
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_question),
                    total=len(df),
                    desc="문제 필터링 중",
                ):
                    if future.result():
                        idx, row = future_to_question[future]
                        filtered_questions.append(
                            (
                                idx,
                                row["question"],
                                {f"{chr(65+i)}": row[chr(65 + i)] for i in range(4)},
                            )
                        )

            logging.info(f"필터링 후 남은 문제: {len(filtered_questions)}개")

            # 문제 변환 및 풀이 생성 (병렬)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # 문제 변환
                future_to_question = {
                    executor.submit(self.transform_question, q[1], q[2]): q
                    for q in filtered_questions
                }

                transformed_questions = []
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_question),
                    total=len(filtered_questions),
                    desc="문제 변환 중",
                ):
                    if future.result():
                        transformed_questions.append(future.result())

                # 풀이 생성
                future_to_transformed = {
                    executor.submit(self.create_problem_solution, q): q
                    for q in transformed_questions
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_transformed),
                    total=len(transformed_questions),
                    desc="풀이 생성 중",
                ):
                    result = future.result()
                    if result:
                        training_pairs.extend(list(result))

                        if len(training_pairs) % (save_interval * 2) == 0:
                            self.save_results(training_pairs)

            # 추가 문제 생성 (병렬)
            needed_pairs = 2000 - len(training_pairs)
            if needed_pairs > 0:
                logging.info(f"추가로 {needed_pairs//2}개의 문제 생성 필요")
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = [
                        executor.submit(self.generate_single_question)
                        for _ in range(needed_pairs // 2)
                    ]

                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=needed_pairs // 2,
                        desc="추가 문제 생성 중",
                    ):
                        result = future.result()
                        if result:
                            training_pairs.extend(result)

                            if len(training_pairs) % (save_interval * 2) == 0:
                                self.save_results(training_pairs)

            # 최종 결과 저장
            if len(training_pairs) > 2000:
                training_pairs = training_pairs[:2000]
            self.save_results(training_pairs, is_final=True)

            # 통계 저장
            stats = {
                "original_questions": len(df),
                "filtered_questions": len(filtered_questions),
                "final_pairs": len(training_pairs),
                "target_achieved": len(training_pairs) == 2000,
                "timestamp": self.timestamp,
            }

            stats_file = self.output_dir / f"processing_stats_{self.timestamp}.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logging.info(f"처리 완료 - 최종 프롬프트-응답 쌍: {len(training_pairs)}개")

        except Exception as e:
            logging.error(f"데이터셋 처리 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API 키를 입력해주세요!")

    try:
        processor = DatasetProcessor(OPENAI_API_KEY)

        # 데이터셋 다운로드 및 처리
        for dataset_type, url in KMMLU_URLS.items():
            logging.info(f"{dataset_type} 데이터셋 처리 시작")

            # 데이터셋 다운로드
            input_file = processor.download_dataset(
                url, processor.data_dir / f"Economics-{dataset_type}.csv"
            )

            # 데이터셋 처리
            processor.process_dataset(input_file)
            logging.info(f"{dataset_type} 데이터셋 처리 완료")

        # 결과 파일 다운로드 (Colab 환경)
        for file_path in processor.output_dir.glob("*"):
            files.download(str(file_path))

        logging.info("모든 처리 완료 및 결과 파일 다운로드 시작")

    except Exception as e:
        logging.error(f"처리 중 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()
