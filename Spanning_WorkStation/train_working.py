import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, BertModel, TrainingArguments
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

# 환경변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 개체 필드 정의
entity_fields = {
    "NAME": "name",
    "BIRTH_DATE": "birth_date",
    "DEATH_DATE": "death_date",
    "OCCUPATION": "occupation",
}

# 라벨 목록 및 id 변환
label_list = ["O"] + list(entity_fields.keys())  # O: non-entity
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}


def get_str(record, field):
    val = record.get(field)
    if pd.isnull(val):
        return ""
    return str(val).strip()


def create_span_labels_with_offsets(text, record, offsets):
    """
    각 개체 필드에 대해 텍스트 내 등장 위치(문자 단위)를 찾고,
    tokenizer의 오프셋(offsets)을 통해 토큰 인덱스 범위로 변환하여
    (start_token_idx, end_token_idx, entity_type) 튜플의 리스트로 반환.
    """
    spans = []
    for tag, field in entity_fields.items():
        entity_value = get_str(record, field)
        if entity_value:
            pattern = re.escape(entity_value)
            match = re.search(pattern, text)
            if match:
                char_start, char_end = match.span()
                token_start, token_end = None, None
                # 오프셋(각 토큰의 [start, end] 문자 위치)에서 해당 토큰 인덱스 찾기
                for i, (s, e) in enumerate(offsets):
                    if s <= char_start < e:
                        token_start = i
                    if s < char_end <= e:
                        token_end = i
                        break
                if token_start is not None and token_end is not None:
                    spans.append((token_start, token_end, tag))
    return spans


def generate_candidate_spans(offsets, max_span_length=10):
    """
    주어진 토큰 오프셋을 바탕으로 가능한 후보 span (시작, 끝) 쌍을 생성.
    (끝 인덱스는 포함하는 형태)
    """
    n = len(offsets)
    candidates = []
    for i in range(n):
        for j in range(i, min(i + max_span_length, n)):
            candidates.append((i, j))
    return candidates


def assign_span_labels(candidate_spans, true_spans):
    """
    각 후보 span에 대해, 실제 true span과 정확히 일치하면 해당 개체 타입(tag)을,
    아니면 "O"를 라벨로 반환.
    """
    labels = []
    true_span_dict = {(s, e): tag for s, e, tag in true_spans}
    for span in candidate_spans:
        if span in true_span_dict:
            labels.append(true_span_dict[span])
        else:
            labels.append("O")
    return labels


class SpanNERDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=256, max_span_length=10):
        self.tokenizer = tokenizer
        self.data = data_list
        self.max_length = max_length
        self.max_span_length = max_span_length
        self.samples = []  # 각 샘플: 토큰화된 입력, 어텐션 마스크, 후보 span, span 라벨(정답)
        self.prepare_data()

    def prepare_data(self):
        for record in self.data:
            text = str(record.get("doc", ""))
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True
            )
            input_ids = encoding["input_ids"].squeeze()  # (seq_len)
            attention_mask = encoding["attention_mask"].squeeze()
            offsets = encoding["offset_mapping"].squeeze().tolist()  # 각 토큰의 [start, end]

            # 실제 개체 span 생성
            true_spans = create_span_labels_with_offsets(text, record, offsets)
            # 후보 span 생성
            candidate_spans = generate_candidate_spans(offsets, self.max_span_length)
            # 후보 span에 라벨 할당 (정확히 일치하면 개체 타입, 아니면 "O")
            span_labels = assign_span_labels(candidate_spans, true_spans)
            span_label_ids = [label2id[label] for label in span_labels]

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "candidate_spans": candidate_spans,
                "span_label_ids": span_label_ids
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': sample["input_ids"],
            'attention_mask': sample["attention_mask"],
            'candidate_spans': sample["candidate_spans"],
            'span_label_ids': torch.tensor(sample["span_label_ids"], dtype=torch.long)
        }


class SpanNERModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(SpanNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        # 후보 span의 시작과 끝 토큰 임베딩을 연결하여 분류
        self.span_classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, candidate_spans, span_label_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        batch_logits = []  # 각 샘플별 span logits 리스트
        losses = []
        # 배치 내 각 샘플에 대해 후보 span마다 예측 수행
        for i, spans in enumerate(candidate_spans):
            token_embeddings = sequence_output[i]  # (seq_len, hidden_size)
            span_embeddings = []
            for (start, end) in spans:
                start_emb = token_embeddings[start]
                end_emb = token_embeddings[end]
                span_emb = torch.cat([start_emb, end_emb], dim=-1)  # (hidden_size*2)
                span_embeddings.append(span_emb)
            span_embeddings = torch.stack(span_embeddings, dim=0)  # (num_spans, hidden_size*2)
            logits = self.span_classifier(span_embeddings)  # (num_spans, num_labels)
            batch_logits.append(logits)
            if span_label_ids is not None:
                labels = span_label_ids[i].to(logits.device)
                loss = F.cross_entropy(logits, labels)
                losses.append(loss)
        if span_label_ids is not None:
            total_loss = torch.stack(losses).mean()
            return total_loss, batch_logits
        return batch_logits


def compute_span_metrics(preds, dataset):
    """
    후보 span 단위의 예측 결과를 모아 전체 F1 (macro) 점수를 계산합니다.
    여기서는 "O" 라벨을 제외한 개체 라벨에 대해 평가합니다.
    """
    all_true = []
    all_pred = []
    for i, sample in enumerate(dataset.samples):
        true_ids = sample["span_label_ids"]
        logits = preds[i]  # (num_spans, num_labels)
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        all_true.extend(true_ids)
        all_pred.extend(pred_ids)
    # O 라벨("O")를 평가에서 제외할 수 있습니다.
    true_labels = [id2label[t] for t in all_true]
    pred_labels = [id2label[p] for p in all_pred]
    # O 라벨을 제외한 라벨 목록
    eval_labels = label_list[1:]
    f1 = f1_score(true_labels, pred_labels, labels=eval_labels, average="macro")
    return {"f1": f1}


# === 메인 실행부 ===
if __name__ == "__main__":
    # 학습/테스트 데이터 로드 (CSV 파일 경로 수정)
    df_train = pd.read_csv(r"C:\junha\Datasets\WikiBio_wikipedia-biography-dataset\train.csv")
    df_test = pd.read_csv(r"C:\junha\Datasets\WikiBio_wikipedia-biography-dataset\val.csv")
    train_data = df_train.to_dict(orient='records')
    test_data = df_test.to_dict(orient='records')

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Span-based Dataset 생성
    train_dataset = SpanNERDataset(train_data, tokenizer)
    test_dataset = SpanNERDataset(test_data, tokenizer)

    config = AutoConfig.from_pretrained(model_name)
    num_labels = len(label2id)
    model = SpanNERModel(model_name, num_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,  # 후보 span 때문에 배치 사이즈를 다소 줄임
        learning_rate=5e-5,
        weight_decay=0.03,
        logging_dir="./logs",
        evaluation_strategy="epoch"
    )


    # collate 함수: 리스트 형태의 candidate_spans를 그대로 배치에 포함시키기 위해 커스터마이징
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        candidate_spans = [item["candidate_spans"] for item in batch]
        span_label_ids = [item["span_label_ids"] for item in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_spans": candidate_spans,
            "span_label_ids": span_label_ids
        }


    # Trainer에 맞추어 compute_loss를 재정의한 커스텀 Trainer 클래스
    from transformers import Trainer


    class SpanNERTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            candidate_spans = inputs["candidate_spans"]
            span_label_ids = inputs["span_label_ids"]
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                candidate_spans=candidate_spans,
                span_label_ids=span_label_ids
            )
            return (loss, logits) if return_outputs else loss


    trainer = SpanNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    print("Training started...")
    trainer.train()
    print("Training completed.")

    model.eval()
    predictions = []
    loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                candidate_spans=batch["candidate_spans"]
            )
            predictions.extend(logits)

    metrics = compute_span_metrics(predictions, test_dataset)
    print("Span-based F1 score:", metrics["f1"])
