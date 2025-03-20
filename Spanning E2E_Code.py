import os
import re
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, BertForQuestionAnswering, Trainer, TrainingArguments

# 환경변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_str(record, field):
    val = record.get(field)
    if pd.isnull(val):
        return ""
    else:
        return str(val).strip()


# CSV 파일의 컬럼명을 엔티티 라벨과 매핑
entity_fields = {
    "NAME": "name",
    "EATTYPE": "eatType",
    "FOOD": "food",
    "PRICERANGE": "priceRange",
    "RATING": "customer rating",
    "AREA": "area",
    "NEAR": "near"
}

# 각 엔티티에 대해 사용할 질문 템플릿
entity_questions = {
    "NAME": "What is the name?",
    "EATTYPE": "What is the eat type?",
    "FOOD": "What is the food?",
    "PRICERANGE": "What is the price range?",
    "RATING": "What is the customer rating?",
    "AREA": "What is the area?",
    "NEAR": "What is near?"
}


def create_qa_instances(data_list):
    """
    각 CSV 레코드에 대해 context(문장)와 미리 정의한 질문, 그리고 해당하는 정답 span 정보를
    SQuAD 형식의 예제로 변환합니다.
    """
    instances = []
    for i, record in enumerate(data_list):
        context = record.get("B열문장", "")
        for label, field in entity_fields.items():
            answer_text = get_str(record, field)
            question = entity_questions[label]
            # 정답이 존재하고 context 내에 있으면 해당 span 정보를 추출
            if answer_text and answer_text in context:
                match = re.search(re.escape(answer_text), context)
                if match:
                    answer_start = match.start()
                    instance = {
                        "id": f"{i}_{label}",
                        "question": question,
                        "context": context,
                        "answers": {"text": [answer_text], "answer_start": [answer_start]}
                    }
                    instances.append(instance)
                else:
                    # answer_text가 context 내에 없으면 빈 정답 처리
                    instance = {
                        "id": f"{i}_{label}",
                        "question": question,
                        "context": context,
                        "answers": {"text": [], "answer_start": []}
                    }
                    instances.append(instance)
            else:
                # 정답이 없는 경우에도 예제로 생성 (빈 정답)
                instance = {
                    "id": f"{i}_{label}",
                    "question": question,
                    "context": context,
                    "answers": {"text": [], "answer_start": []}
                }
                instances.append(instance)
    return instances


class QADataset(Dataset):
    """
    각 질문-답변 예제를 토크나이저로 전처리하여 모델 학습에 적합한 feature로 변환합니다.
    """

    def __init__(self, examples, tokenizer, max_length=384, doc_stride=128):
        self.features = []
        for ex in examples:
            # 한 예제에 대해 question과 context를 함께 토크나이즈
            inputs = tokenizer(
                ex["question"],
                ex["context"],
                truncation="only_second",
                max_length=max_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            # 여기서는 overflow된 경우 첫 번째 청크만 사용 (필요시 추가 처리 가능)
            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            offset_mapping = inputs["offset_mapping"][0]
            # token_type_ids: 0은 질문, 1은 context (BERT의 경우)
            token_type_ids = inputs["token_type_ids"][0]

            # 정답이 있는 경우 정답의 시작/끝 token index를 찾음
            if ex["answers"]["text"]:
                answer_text = ex["answers"]["text"][0]
                answer_start_char = ex["answers"]["answer_start"][0]
                answer_end_char = answer_start_char + len(answer_text)
            else:
                answer_text = ""
                answer_start_char = None
                answer_end_char = None

            # context에 해당하는 token 범위: token_type_ids가 1인 부분
            token_start_index = 0
            while token_start_index < len(token_type_ids) and token_type_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while token_end_index >= 0 and token_type_ids[token_end_index] != 1:
                token_end_index -= 1

            # 정답이 span 내에 포함되지 않으면 0으로 지정 (모델이 no-answer를 예측하도록 함)
            if answer_start_char is None or not (
                    offset_mapping[token_start_index][0] <= answer_start_char and offset_mapping[token_end_index][
                1] >= answer_end_char):
                start_position = 0
                end_position = 0
            else:
                # 정답의 token 시작 인덱스 찾기
                for idx in range(token_start_index, token_end_index + 1):
                    if offset_mapping[idx][0] <= answer_start_char < offset_mapping[idx][1]:
                        start_position = idx
                        break
                # 정답의 token 종료 인덱스 찾기
                for idx in range(token_end_index, token_start_index - 1, -1):
                    if offset_mapping[idx][0] < answer_end_char <= offset_mapping[idx][1]:
                        end_position = idx
                        break

            self.features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "start_positions": start_position,
                "end_positions": end_position,
                "id": ex["id"],
                "question": ex["question"],
                "context": ex["context"],
                "answers": ex["answers"]
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(feature["token_type_ids"], dtype=torch.long),
            "start_positions": torch.tensor(feature["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(feature["end_positions"], dtype=torch.long),
            "id": feature["id"],
            "question": feature["question"],
            "context": feature["context"],
            "answers": feature["answers"]
        }


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer):
    """
    모델 예측 결과(시작/종료 logits)를 토큰 ID로부터 실제 문자열 정답으로 변환합니다.
    """
    all_start_logits, all_end_logits = raw_predictions
    predictions = {}
    for i, feature in enumerate(features):
        input_ids = feature["input_ids"]
        # 예측된 시작, 종료 인덱스 선택
        start_index = int(np.argmax(all_start_logits[i]))
        end_index = int(np.argmax(all_end_logits[i]))
        # 예측된 정답 문자열 (start_index가 0이면 no-answer)
        if start_index == 0:
            pred_answer = ""
        else:
            pred_answer = tokenizer.decode(input_ids[start_index:end_index + 1], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)
        predictions[feature["id"]] = {
            "question": feature["question"],
            "context": feature["context"],
            "ground_truth": feature["answers"]["text"][0] if feature["answers"]["text"] else "",
            "prediction": pred_answer
        }
    return predictions


if __name__ == "__main__":
    # CSV 파일에서 데이터 읽기 (경로와 시트명은 실제 파일에 맞게 수정)
    df_train = pd.read_csv(r"C:/junha/Datasets/E2E/filename_updated.csv")
    df_test = pd.read_csv(r"C:/junha/Datasets/E2E/filename_updated_test.csv")

    # 각 데이터프레임을 dict 리스트로 변환 (각 행이 하나의 record)
    train_data = df_train.to_dict(orient='records')
    test_data = df_test.to_dict(orient='records')

    # 각 record를 질문-답변 예제로 변환 (엔티티별로 하나씩)
    train_instances = create_qa_instances(train_data)
    test_instances = create_qa_instances(test_data)

    # 모델과 토크나이저 로드 (다국어 BERT 사용)
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=config)

    # 학습, 테스트 데이터셋 생성
    train_dataset = QADataset(train_instances, tokenizer, max_length=384, doc_stride=128)
    test_dataset = QADataset(test_instances, tokenizer, max_length=384, doc_stride=128)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="./results_qa",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir="./logs_qa",
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    print("Training started...")
    trainer.train()
    print("Training completed.")

    # 테스트 데이터에 대해 예측 수행
    print("Predicting on test dataset...")
    predictions_raw = trainer.predict(test_dataset)
    # predictions_raw.predictions는 (start_logits, end_logits)
    # 테스트 데이터셋의 모든 feature(질문-답변 예제)는 trainer의 내부 순서와 동일함
    test_features = [test_dataset[i] for i in range(len(test_dataset))]
    predictions = postprocess_qa_predictions(test_instances, test_features, predictions_raw.predictions, tokenizer)

    # 문서별(엔티티별) 예측 결과 출력
    print("Test Predictions:")
    for ex_id, result in predictions.items():
        print(f"ID: {ex_id}")
        print(f"Question: {result['question']}")
        print(f"Context: {result['context']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Prediction: {result['prediction']}")
        print("-" * 50)

    # 예측 결과를 JSON 파일로 저장
    with open("qa_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print("Predictions saved to qa_predictions.json")
