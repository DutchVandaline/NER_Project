import os
import json
import re
import torch
import numpy as np
import pandas as pd
import string
from collections import Counter
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_str(record, field):
    val = record.get(field)
    if pd.isnull(val):
        return ""
    else:
        return str(val).strip()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

label2id = {
    "O": 0,
    "B-NAME": 1, "I-NAME": 2,
    "B-EATTYPE": 3, "I-EATTYPE": 4,
    "B-FOOD": 5, "I-FOOD": 6,
    "B-PRICERANGE": 7, "I-PRICERANGE": 8,
    "B-RATING": 9, "I-RATING": 10,
    "B-AREA": 11, "I-AREA": 12,
    "B-NEAR": 13, "I-NEAR": 14
}
id2label = {v: k for k, v in label2id.items()}

def create_bio_labels_with_offsets(text, tokens, offsets, record):
    labels = ["O"] * len(tokens)
    entity_fields = {
        "NAME": "name",
        "EATTYPE": "eatType",
        "FOOD": "food",
        "PRICERANGE": "priceRange",
        "RATING": "customer rating",
        "AREA": "area",
        "NEAR": "near"
    }
    for tag, field in entity_fields.items():
        try:
            entity_value = str(record.get(field)).strip() if record.get(field) is not None else ""
        except Exception:
            entity_value = ""
        if entity_value:
            pattern = re.escape(entity_value)
            matches = re.finditer(pattern, text)
            for match in matches:
                ent_start, ent_end = match.span()
                b_assigned = False
                for i, (token_start, token_end) in enumerate(offsets):
                    if token_start == token_end == 0:
                        continue
                    if token_end > ent_start and token_start < ent_end:
                        if not b_assigned:
                            labels[i] = f"B-{tag}"
                            b_assigned = True
                        else:
                            labels[i] = f"I-{tag}"
                break
    return labels

def extract_entities(labels, tokens, tag):
    entities = []
    current_entity_tokens = []
    for i, (label, token) in enumerate(zip(labels, tokens)):
        token_clean = token[2:] if token.startswith("##") else token
        if label == f"B-{tag}":
            if current_entity_tokens:
                entities.append("".join(current_entity_tokens))
            current_entity_tokens = [token_clean]
        elif label == f"I-{tag}" and current_entity_tokens:
            if token.startswith("##"):
                current_entity_tokens.append(token_clean)
            else:
                if token in string.punctuation:
                    current_entity_tokens.append(token_clean)
                else:
                    if current_entity_tokens and current_entity_tokens[-1][-1] in string.punctuation:
                        current_entity_tokens.append(token_clean)
                    else:
                        current_entity_tokens.append(" " + token_clean)
        else:
            if current_entity_tokens:
                entities.append("".join(current_entity_tokens))
                current_entity_tokens = []
    if current_entity_tokens:
        entities.append("".join(current_entity_tokens))
    return " ".join(entities)

def labeling(data_list):
    all_input_ids = []
    all_attention_masks = []
    all_label_ids = []
    sentences_iob = []
    for record in data_list:
        text = record.get("B열문장", "")
        encoding = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
            return_offsets_mapping=True
        )
        input_ids_list = encoding["input_ids"].squeeze().tolist()
        attention_mask_list = encoding["attention_mask"].squeeze().tolist()
        offsets = encoding["offset_mapping"].squeeze().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
        bio_labels = create_bio_labels_with_offsets(text, tokens, offsets, record)
        label_ids = [label2id.get(lbl, label2id["O"]) for lbl in bio_labels]
        all_input_ids.append(input_ids_list)
        all_attention_masks.append(attention_mask_list)
        all_label_ids.append(label_ids)
        sentences_iob.append({"tokens": tokens, "labels": bio_labels})
    return all_input_ids, all_attention_masks, all_label_ids, sentences_iob

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_masks, label_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.label_ids[idx], dtype=torch.long)
        }

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []
    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            true_seq.append(id2label[labels[i][j]])
            pred_seq.append(id2label[predictions[i][j]])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)
    f1 = f1_score(true_labels, pred_labels)
    return {"f1": f1}

def compute_macro_metrics_per_document(test_input_ids, test_label_ids, predictions, attention_masks, tokenizer, id2label):
    entity_labels = ["O", "NAME", "EATTYPE", "FOOD", "PRICERANGE", "RATING", "AREA", "NEAR"]
    metrics_per_entity = {label: {"precisions": [], "recalls": [], "f1s": []} for label in entity_labels}

    def map_bio_to_entity(label):
        if label == "O":
            return "O"
        else:
            return label.split("-")[-1]

    for i in range(len(test_input_ids)):
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids[i])
        true_token_labels = [id2label[label] for label in test_label_ids[i]]
        pred_token_labels = [id2label[p] for p in predictions[i]]
        mask = attention_masks[i]
        filtered_true = []
        filtered_pred = []
        for tok, att, t, p in zip(tokens, mask, true_token_labels, pred_token_labels):
            if att == 1 and tok not in {"[CLS]", "[SEP]", "[PAD]"}:
                filtered_true.append(t)
                filtered_pred.append(p)
        mapped_true = [map_bio_to_entity(lab) for lab in filtered_true]
        mapped_pred = [map_bio_to_entity(lab) for lab in filtered_pred]
        cm = confusion_matrix(mapped_true, mapped_pred, labels=entity_labels)
        for idx, entity in enumerate(entity_labels):
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
    for entity in entity_labels:
        prec_avg = np.mean(metrics_per_entity[entity]["precisions"]) if metrics_per_entity[entity]["precisions"] else 0.0
        rec_avg = np.mean(metrics_per_entity[entity]["recalls"]) if metrics_per_entity[entity]["recalls"] else 0.0
        f1_avg = np.mean(metrics_per_entity[entity]["f1s"]) if metrics_per_entity[entity]["f1s"] else 0.0
        macro_metrics[entity] = {"precision": prec_avg, "recall": rec_avg, "f1": f1_avg}
        print(f"Entity: {entity} - Macro Precision: {prec_avg:.4f}, Macro Recall: {rec_avg:.4f}, Macro F1: {f1_avg:.4f}")
    return macro_metrics

def exact_matching_accuracy_per_document(test_data, test_input_ids, predictions, tokenizer, id2label):
    entity_types = ["NAME", "EATTYPE", "FOOD", "PRICERANGE", "RATING", "AREA", "NEAR"]
    overall_metrics = {}
    for entity in entity_types:
        overall_metrics[entity] = {"correct": 0, "total": len(test_data), "accuracy": 0.0}
    for idx, record in enumerate(test_data):
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids[idx])
        pred_labels = [id2label[p] for p in predictions[idx][:len(tokens)]]
        for entity, field in zip(entity_types, ["name", "eatType", "food", "priceRange", "customer rating", "area", "near"]):
            true_entity = get_str(record, field)
            pred_entity = extract_entities(pred_labels, tokens, entity).strip()
            if true_entity == pred_entity:
                overall_metrics[entity]["correct"] += 1
    print("\nEntity-level Exact Matching Accuracy (Document-level):")
    for entity in entity_types:
        overall_metrics[entity]["accuracy"] = overall_metrics[entity]["correct"] / overall_metrics[entity]["total"]
        print(f"{entity} -> Accuracy: {overall_metrics[entity]['accuracy']:.4f}")
    return overall_metrics

def print_text_and_entity_predictions(test_data, test_input_ids, predictions, tokenizer, id2label):
    print("Document-wise Entity Prediction Results:")
    for idx, record in enumerate(test_data):
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids[idx])
        pred_labels = [id2label[p] for p in predictions[idx][:len(tokens)]]
        true_entities = {
            "NAME": get_str(record, "name"),
            "EATTYPE": get_str(record, "eatType"),
            "FOOD": get_str(record, "food"),
            "PRICERANGE": get_str(record, "priceRange"),
            "RATING": get_str(record, "customer rating"),
            "AREA": get_str(record, "area"),
            "NEAR": get_str(record, "near"),
        }
        pred_entities = {tag: extract_entities(pred_labels, tokens, tag).strip() for tag in true_entities}
        text = get_str(record, "질문수정")
        print(f"Text: {text}")
        for tag in true_entities:
            true_val = true_entities[tag] if true_entities[tag] else None
            pred_val = pred_entities[tag] if pred_entities[tag] else None
            print(f"True {tag}: {true_val} | Predicted {tag}: {pred_val}")
        print("-" * 50)

def save_all_evaluation_excel(test_data, test_input_ids, test_label_ids, predictions, tokenizer, id2label,
                              filename="all_evaluation.xlsx", overall_metrics=None):
    rows = []
    n = len(test_data)
    for i in range(n):
        record = test_data[i]
        text = get_str(record, "질문수정")
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids[i])
        true_token_labels = [id2label[label_id] for label_id in test_label_ids[i]]
        pred_token_labels = [id2label[p] for p in predictions[i][:len(tokens)]]
        bio_pairs = []
        for tok, lab in zip(tokens, true_token_labels):
            if tok in ["[CLS]", "[SEP]"]:
                continue
            bio_pairs.append(f"{tok}/{lab}")
        bio_str = "\n".join(bio_pairs)
        true_name = get_str(record, "name")
        true_eattype = get_str(record, "eatType")
        true_food = get_str(record, "food")
        true_pricerange = get_str(record, "priceRange")
        true_rating = get_str(record, "customer rating")
        true_area = get_str(record, "area")
        true_near = get_str(record, "near")
        pred_name = extract_entities(pred_token_labels, tokens, "NAME").strip()
        pred_eattype = extract_entities(pred_token_tokens=pred_token_labels, tokens=tokens, tag="EATTYPE").strip() if "EATTYPE" in id2label.values() else extract_entities(pred_token_labels, tokens, "EATTYPE").strip()
        pred_eattype = extract_entities(pred_token_labels, tokens, "EATTYPE").strip()
        pred_food = extract_entities(pred_token_labels, tokens, "FOOD").strip()
        pred_pricerange = extract_entities(pred_token_labels, tokens, "PRICERANGE").strip()
        pred_rating = extract_entities(pred_token_labels, tokens, "RATING").strip()
        pred_area = extract_entities(pred_token_labels, tokens, "AREA").strip()
        pred_near = extract_entities(pred_token_labels, tokens, "NEAR").strip()
        if tokens[0] == "[CLS]" and tokens[-1] == "[SEP]":
            t_tokens = tokens[1:-1]
            t_true = true_token_labels[1:-1]
            t_pred = pred_token_labels[1:-1]
        else:
            t_tokens = tokens
            t_true = true_token_labels
            t_pred = pred_token_labels
        token_accuracy = np.mean([1 if t == p else 0 for t, p in zip(t_true, t_pred)])
        token_eval = f"Token Accuracy: {token_accuracy * 100:.2f}%"
        row = {
            "Text": text,
            "BIO Labeling": bio_str,
            "True NAME / Predicted NAME": f"True: {true_name} | Predicted: {pred_name}",
            "True EATTYPE / Predicted EATTYPE": f"True: {true_eattype} | Predicted: {pred_eattype}",
            "True FOOD / Predicted FOOD": f"True: {true_food} | Predicted: {pred_food}",
            "True PRICERANGE / Predicted PRICERANGE": f"True: {true_pricerange} | Predicted: {pred_pricerange}",
            "True RATING / Predicted RATING": f"True: {true_rating} | Predicted: {pred_rating}",
            "True AREA / Predicted AREA": f"True: {true_area} | Predicted: {pred_area}",
            "True NEAR / Predicted NEAR": f"True: {true_near} | Predicted: {pred_near}",
            "Token Evaluation": token_eval
        }
        rows.append(row)
    record_df = pd.DataFrame(rows)
    overall_list = []
    if overall_metrics is None:
        overall_metrics = exact_matching_accuracy_per_document(test_data, test_input_ids, predictions, tokenizer, id2label)
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

def save_confusion_matrix_images(test_input_ids, test_label_ids, predictions, attention_masks, tokenizer, id2label,
                                 filename_base="token_confusion_matrix"):
    def map_bio_to_entity(label):
        if label == "O":
            return "O"
        else:
            return label.split("-")[-1]
    all_true = []
    all_pred = []
    for i in range(len(test_input_ids)):
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids[i])
        true_labels = [id2label[label] for label in test_label_ids[i]]
        pred_labels = [id2label[p] for p in predictions[i]]
        mask = attention_masks[i]
        for tok, att, t, p in zip(tokens, mask, true_labels, pred_labels):
            if att == 1 and tok not in {"[CLS]", "[SEP]", "[PAD]"}:
                all_true.append(map_bio_to_entity(t))
                all_pred.append(map_bio_to_entity(p))
    classes = ["O", "NAME", "EATTYPE", "FOOD", "PRICERANGE", "RATING", "AREA", "NEAR"]
    cm = confusion_matrix(all_true, all_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Token-level Confusion Matrix (Count-based)")
    plt.tight_layout()
    plt.savefig(f"{filename_base}_count.png")
    plt.close()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Token-level Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{filename_base}_normalized.png")
    plt.close()
    print(f"Confusion matrix images saved to {filename_base}_count.png and {filename_base}_normalized.png")

if __name__ == "__main__":
    df_train = pd.read_csv(r"C:/junha/Datasets/E2E/filename_updated.csv")
    df_test = pd.read_csv(r"C:/junha/Datasets/E2E/filename_updated_test.csv")
    train_data = df_train.to_dict(orient='records')
    test_data = df_test.to_dict(orient='records')
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_input_ids, train_attention_masks, train_label_ids, train_sentences_iob = labeling(train_data)
    test_input_ids, test_attention_masks, test_label_ids, test_sentences_iob = labeling(test_data)
    train_output_file = "train_labeling_results.txt"
    with open(train_output_file, "w", encoding="utf-8") as f:
        for sent_dict in train_sentences_iob:
            tokens = sent_dict["tokens"]
            labels = sent_dict["labels"]
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")
    print(f"Train 라벨링 결과 저장 완료: {train_output_file}")
    test_output_file = "test_labeling_results.txt"
    with open(test_output_file, "w", encoding="utf-8") as f:
        for sent_dict in test_sentences_iob:
            tokens = sent_dict["tokens"]
            labels = sent_dict["labels"]
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")
    print(f"Test 라벨링 결과 저장 완료: {test_output_file}")
    config = AutoConfig.from_pretrained(model_name, num_labels=len(label2id))
    model = BertForTokenClassification.from_pretrained(model_name, config=config)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.03,
        logging_dir="./logs"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=NERDataset(train_input_ids, train_attention_masks, train_label_ids),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Training started...")
    trainer.train()
    print("Training completed.")
    preds, _, _ = trainer.predict(NERDataset(test_input_ids, test_attention_masks, test_label_ids))
    predictions = np.argmax(preds, axis=2)
    print_text_and_entity_predictions(
        test_data=test_data,
        test_input_ids=test_input_ids,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=id2label
    )
    macro_metrics = compute_macro_metrics_per_document(
        test_input_ids=test_input_ids,
        test_label_ids=test_label_ids,
        predictions=predictions,
        attention_masks=test_attention_masks,
        tokenizer=tokenizer,
        id2label=id2label
    )
    overall_metrics = exact_matching_accuracy_per_document(
        test_data=test_data,
        test_input_ids=test_input_ids,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=id2label
    )
    save_all_evaluation_excel(
        test_data=test_data,
        test_input_ids=test_input_ids,
        test_label_ids=test_label_ids,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=id2label,
        filename="all_evaluation.xlsx",
        overall_metrics=overall_metrics
    )
    save_confusion_matrix_images(
        test_input_ids=test_input_ids,
        test_label_ids=test_label_ids,
        predictions=predictions,
        attention_masks=test_attention_masks,
        tokenizer=tokenizer,
        id2label=id2label,
        filename_base="token_confusion_matrix"
    )
    test_true_file = "test_true_labels.txt"
    test_pred_file = "test_pred_labels.txt"
    with open(test_true_file, "w", encoding="utf-8") as f_true, \
         open(test_pred_file, "w", encoding="utf-8") as f_pred:
        for i in range(len(test_data)):
            tokens_ids = test_input_ids[i]
            true_ids = test_label_ids[i]
            pred_ids = predictions[i]
            tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
            true_labels = [id2label[t_id] for t_id in true_ids]
            pred_labels = [id2label[p_id] for p_id in pred_ids]
            for token, t_lab, p_lab in zip(tokens, true_labels, pred_labels):
                if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                    f_true.write(f"{token}\t{t_lab}\n")
                    f_pred.write(f"{token}\t{p_lab}\n")
            f_true.write("\n")
            f_pred.write("\n")
    print(f"실제 라벨 파일 저장 완료: {test_true_file}")
    print(f"예측 라벨 파일 저장 완료: {test_pred_file}")
