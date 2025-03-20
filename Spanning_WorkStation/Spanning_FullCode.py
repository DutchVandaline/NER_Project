import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, BertModel, TrainingArguments
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 환경변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

entity_fields = {
    "NAME": "name",
    "BIRTH_DATE": "birth_date",
    "DEATH_DATE": "death_date",
    "OCCUPATION": "occupation",
}

# 라벨 목록 및 id 변환 (O 포함)
label_list = ["O"] + list(entity_fields.keys())  # ["O", "NAME", "BIRTH_DATE", "DEATH_DATE", "OCCUPATION"]
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
                # 오프셋에서 해당 토큰 인덱스 찾기
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
    (끝 인덱스 포함)
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
        self.samples = []  # 각 샘플: 토큰화된 입력, 어텐션 마스크, 후보 span, span 정답 라벨
        self.prepare_data()

    def prepare_data(self):
        for record in self.data:
            # 원문 텍스트는 "doc" 컬럼 사용
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


# --- Evaluation 및 Helper 함수 ---

def span_to_text(tokens, span):
    """
    tokens: tokenizer.convert_ids_to_tokens() 결과 리스트
    span: (start, end) 튜플 (끝 인덱스 포함)
    """
    span_tokens = tokens[span[0]:span[1] + 1]
    text = ""
    for token in span_tokens:
        if token.startswith("##"):
            text += token[2:]
        else:
            if text:
                text += " " + token
            else:
                text += token
    return text.strip()


def compute_span_metrics(preds, dataset, id2label, label_list):
    """
    전체 후보 span 단위의 예측 결과에 대해, "O" 라벨을 제외한 개체 라벨에 대한 macro F1 score 계산
    """
    all_true = []
    all_pred = []
    for i, sample in enumerate(dataset.samples):
        true_ids = sample['span_label_ids']  # 리스트(int)
        logits = preds[i]  # (num_candidate_spans, num_labels)
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        all_true.extend(true_ids)
        all_pred.extend(pred_ids)
    true_labels = [id2label[t] for t in all_true]
    pred_labels = [id2label[p] for p in all_pred]
    # "O" 라벨을 제외하고 평가 (macro)
    eval_labels = label_list[1:]
    f1 = f1_score(true_labels, pred_labels, labels=eval_labels, average="macro")
    return {"f1": f1}


def compute_macro_metrics_per_document_span(dataset, predictions, id2label, label_list):
    """
    각 문서(샘플)별 후보 span 단위의 예측 결과를 모아, 각 라벨별 Macro Precision, Recall, F1 계산
    """
    metrics_per_entity = {label: {"precisions": [], "recalls": [], "f1s": []} for label in label_list}
    for i, sample in enumerate(dataset.samples):
        true_ids = sample["span_label_ids"]
        true_labels = [id2label[x] for x in true_ids]
        logits = predictions[i]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        pred_labels = [id2label[x] for x in pred_ids]
        cm = confusion_matrix(true_labels, pred_labels, labels=label_list)
        for idx, entity in enumerate(label_list):
            if cm[idx, :].sum() == 0 and cm[:, idx].sum() == 0:
                precision = recall = f1 = 1.0
            else:
                TP = cm[idx, idx]
                FP = cm[:, idx].sum() - TP
                FN = cm[idx, :].sum() - TP
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            metrics_per_entity[entity]["precisions"].append(precision)
            metrics_per_entity[entity]["recalls"].append(recall)
            metrics_per_entity[entity]["f1s"].append(f1)
    macro_metrics = {}
    for entity in label_list:
        prec_avg = np.mean(metrics_per_entity[entity]["precisions"])
        rec_avg = np.mean(metrics_per_entity[entity]["recalls"])
        f1_avg = np.mean(metrics_per_entity[entity]["f1s"])
        macro_metrics[entity] = {"precision": prec_avg, "recall": rec_avg, "f1": f1_avg}
        print(f"Entity: {entity} - Macro Precision: {prec_avg:.4f}, Macro Recall: {rec_avg:.4f}, Macro F1: {f1_avg:.4f}")
    return macro_metrics


def exact_matching_accuracy_per_document_span(test_data, dataset, predictions, tokenizer, id2label):
    """
    문서 단위 정확도 계산:
    각 문서에서 각 개체 필드(여기서는 "NAME", "BIRTH_DATE", "DEATH_DATE", "OCCUPATION")에 대해,
    후보 span 중 predicted label이 해당 개체인 것 중, 실제 토큰 텍스트(후처리 후)가 ground truth와 정확히 일치하면 정답으로 간주.
    """
    entity_types = list(entity_fields.keys())  # ["NAME", "BIRTH_DATE", "DEATH_DATE", "OCCUPATION"]
    fields = [entity_fields[e] for e in entity_types]  # ["name", "birth_date", "death_date", "occupation"]
    overall_metrics = {entity: {"correct": 0, "total": len(test_data), "accuracy": 0.0} for entity in entity_types}

    for idx, record in enumerate(test_data):
        input_ids = dataset.samples[idx]['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        candidate_spans = dataset.samples[idx]['candidate_spans']
        logits = predictions[idx]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        for entity, field in zip(entity_types, fields):
            true_entity = get_str(record, field).strip()
            found = False
            for span, pred_label_id in zip(candidate_spans, pred_ids):
                pred_label = id2label[pred_label_id]
                if pred_label == entity:
                    span_text = span_to_text(tokens, span)
                    if span_text.strip() == true_entity:
                        found = True
                        break
            if found:
                overall_metrics[entity]["correct"] += 1
    print("\nEntity-level Exact Matching Accuracy (Document-level):")
    for entity in entity_types:
        overall_metrics[entity]["accuracy"] = overall_metrics[entity]["correct"] / overall_metrics[entity]["total"]
        print(f"{entity} -> Accuracy: {overall_metrics[entity]['accuracy']:.4f}")
    return overall_metrics


def print_text_and_entity_predictions_span(test_data, dataset, predictions, tokenizer, id2label):
    """
    각 문서별로 원문(doc 컬럼), 각 개체 필드의 ground truth와 predicted span(텍스트)을 출력합니다.
    """
    print("Document-wise Entity Prediction Results (Span-based):")
    for idx, record in enumerate(test_data):
        input_ids = dataset.samples[idx]['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        candidate_spans = dataset.samples[idx]['candidate_spans']
        logits = predictions[idx]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        # 각 entity에 대해 첫 번째로 예측된 candidate span의 텍스트 선택
        pred_entities = {entity: None for entity in entity_fields.keys()}
        for span, pred_label_id in zip(candidate_spans, pred_ids):
            pred_label = id2label[pred_label_id]
            if pred_label != "O" and pred_label in pred_entities and pred_entities[pred_label] is None:
                pred_entities[pred_label] = span_to_text(tokens, span).strip()
        true_entities = {
            "NAME": get_str(record, "name").strip(),
            "BIRTH_DATE": get_str(record, "birth_date").strip(),
            "DEATH_DATE": get_str(record, "death_date").strip(),
            "OCCUPATION": get_str(record, "occupation").strip()
        }
        # 원문 텍스트는 "doc" 컬럼 사용
        text = get_str(record, "doc")
        print(f"Text: {text}")
        for tag in entity_fields.keys():
            true_val = true_entities[tag] if true_entities[tag] else None
            pred_val = pred_entities[tag] if pred_entities[tag] else None
            print(f"True {tag}: {true_val} | Predicted {tag}: {pred_val}")
        print("-" * 50)


def save_all_evaluation_excel_span(test_data, dataset, predictions, tokenizer, id2label,
                                   filename="all_evaluation_span.xlsx", overall_metrics=None):
    """
    각 문서별로 평가 결과(원문, candidate span 정보, 각 필드의 ground truth와 예측값)를 엑셀 파일로 저장합니다.
    """
    rows = []
    n = len(test_data)
    for i in range(n):
        record = test_data[i]
        input_ids = dataset.samples[i]['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        candidate_spans = dataset.samples[i]['candidate_spans']
        logits = predictions[i]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        span_info = []
        for span, p_id in zip(candidate_spans, pred_ids):
            span_text = span_to_text(tokens, span)
            label = id2label[p_id]
            span_info.append(f"{span_text} ({label})")
        span_info_str = "\n".join(span_info)
        true_name = get_str(record, "name")
        true_birth_date = get_str(record, "birth_date")
        true_death_date = get_str(record, "death_date")
        true_occupation = get_str(record, "occupation")

        # 각 entity별 예측: 첫 번째로 해당 entity로 예측된 candidate span의 텍스트 사용
        pred_entities = {entity: "" for entity in entity_fields.keys()}
        for span, p_id in zip(candidate_spans, pred_ids):
            pred_label = id2label[p_id]
            if pred_label in pred_entities and not pred_entities[pred_label]:
                pred_entities[pred_label] = span_to_text(tokens, span)

        row = {
            "Text": get_str(record, "doc"),
            "Span Info": span_info_str,
            "True NAME / Predicted NAME": f"True: {true_name} | Predicted: {pred_entities['NAME']}",
            "True BIRTH_DATE / Predicted BIRTH_DATE": f"True: {true_birth_date} | Predicted: {pred_entities['BIRTH_DATE']}",
            "True DEATH_DATE / Predicted DEATH_DATE": f"True: {true_death_date} | Predicted: {pred_entities['DEATH_DATE']}",
            "True OCCUPATION / Predicted OCCUPATION": f"True: {true_occupation} | Predicted: {pred_entities['OCCUPATION']}"
        }
        rows.append(row)
    record_df = pd.DataFrame(rows)
    overall_list = []
    if overall_metrics is None:
        overall_metrics = exact_matching_accuracy_per_document_span(test_data, dataset, predictions, tokenizer, id2label)
    for entity, metrics in overall_metrics.items():
        overall_list.append({
            "Entity": entity,
            "Correct": metrics["correct"],
            "Total": metrics["total"],
            "Accuracy": round(metrics["accuracy"], 4)
        })
    overall_df = pd.DataFrame(overall_list)
    with pd.ExcelWriter(filename) as writer:
        record_df.to_excel(writer, sheet_name="Record Level Evaluation", index=False)
        overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
    print("All evaluation results saved to", filename)


def save_confusion_matrix_images_span(dataset, predictions, id2label, label_list,
                                      filename_base="span_confusion_matrix"):
    """
    전체 후보 span에 대해, 정답 라벨과 예측 라벨의 혼동 행렬(카운트 및 정규화)을 이미지 파일로 저장합니다.
    """
    all_true = []
    all_pred = []
    for i, sample in enumerate(dataset.samples):
        true_ids = sample['span_label_ids']
        logits = predictions[i]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        all_true.extend(true_ids)
        all_pred.extend(pred_ids)
    true_labels = [id2label[t] for t in all_true]
    pred_labels = [id2label[p] for p in all_pred]
    classes = label_list
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Span-level Confusion Matrix (Count-based)")
    plt.tight_layout()
    plt.savefig(f"{filename_base}_count.png")
    plt.close()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Span-level Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{filename_base}_normalized.png")
    plt.close()
    print(f"Confusion matrix images saved to {filename_base}_count.png and {filename_base}_normalized.png")


# === 메인 실행부 ===
if __name__ == "__main__":
    # 학습/테스트 데이터 로드 (CSV 파일 경로 수정)
    df_train = pd.read_csv(r"C:/junha/Datasets/E2E/filename_updated.csv")
    df_test = pd.read_csv(r"C:/junha/Datasets/E2E/filename_updated_test.csv")
    train_data = df_train.to_dict(orient='records')
    test_data = df_test.to_dict(orient='records')

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Span-based Dataset 생성 (원문은 "doc" 컬럼 사용)
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

    # collate 함수: candidate_spans 리스트를 그대로 배치에 포함시키기 위해 커스터마이징
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

    # 평가 및 예측 (DataLoader 활용)
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

    metrics = compute_span_metrics(predictions, test_dataset, id2label, label_list)
    macro_metrics = compute_macro_metrics_per_document_span(test_dataset, predictions, id2label, label_list)
    doc_exact = exact_matching_accuracy_per_document_span(test_data, test_dataset, predictions, tokenizer, id2label)
    print_text_and_entity_predictions_span(test_data, test_dataset, predictions, tokenizer, id2label)
    save_all_evaluation_excel_span(test_data, test_dataset, predictions, tokenizer, id2label,
                                   filename="all_evaluation_span.xlsx")
    save_confusion_matrix_images_span(test_dataset, predictions, id2label, label_list,
                                      filename_base="span_confusion_matrix")

    print("Span-based F1 score:", metrics["f1"])
