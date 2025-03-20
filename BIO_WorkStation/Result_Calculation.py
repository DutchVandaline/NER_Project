import numpy as np
import pandas as pd
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from BIO_WorkStation.NER_Labeling import id2label, extract_entities


def get_str(record, field):
    val = record.get(field)
    if pd.isnull(val):
        return ""
    else:
        return str(val).strip()

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


def compute_macro_metrics_per_document(test_input_ids, test_label_ids, predictions, attention_masks, tokenizer,
                                       id2label):
    entity_labels = ["O", "NAME", "BIRTHDATE", "DEATHDATE", "OCCUPATION"]
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
        prec_avg = np.mean(metrics_per_entity[entity]["precisions"]) if metrics_per_entity[entity][
            "precisions"] else 0.0
        rec_avg = np.mean(metrics_per_entity[entity]["recalls"]) if metrics_per_entity[entity]["recalls"] else 0.0
        f1_avg = np.mean(metrics_per_entity[entity]["f1s"]) if metrics_per_entity[entity]["f1s"] else 0.0
        macro_metrics[entity] = {"precision": prec_avg, "recall": rec_avg, "f1": f1_avg}
        print(
            f"Entity: {entity} - Macro Precision: {prec_avg:.4f}, Macro Recall: {rec_avg:.4f}, Macro F1: {f1_avg:.4f}")
    return macro_metrics


def exact_matching_accuracy_per_document(test_data, test_input_ids, predictions, tokenizer, id2label):
    # 엔티티 종류와 record 필드 이름 변경 (familyFriendly 제거)
    entity_types = ["NAME", "BIRTHDATE", "DEATHDATE", "OCCUPATION"]
    overall_metrics = {}
    for entity in entity_types:
        overall_metrics[entity] = {"correct": 0, "total": len(test_data), "accuracy": 0.0}
    for idx, record in enumerate(test_data):
        tokens = tokenizer.convert_ids_to_tokens(test_input_ids[idx])
        pred_labels = [id2label[p] for p in predictions[idx][:len(tokens)]]
        for entity, field in zip(entity_types,
                                 ["name", "birth_date", "death_date", "occupation"]):
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
            "BIRTHDATE": get_str(record, "birth_date"),
            "DEATHDATE": get_str(record, "death_date"),
            "OCCUPATION": get_str(record, "occupation"),
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
        true_birthdate = get_str(record, "birth_date")
        true_deathdate = get_str(record, "death_date")
        true_occupation = get_str(record, "occupation")

        pred_name = extract_entities(pred_token_labels, tokens, "NAME").strip()
        pred_birthdate = extract_entities(pred_token_labels, tokens, "BIRTHDATE").strip()
        pred_deathdate = extract_entities(pred_token_labels, tokens, "DEATHDATE").strip()
        pred_occupation = extract_entities(pred_token_labels, tokens, "OCCUPATION").strip()

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
            "True BIRTHDATE / Predicted BIRTHDATE": f"True: {true_birthdate} | Predicted: {pred_birthdate}",
            "True DEATHDATE / Predicted DEATHDATE": f"True: {true_deathdate} | Predicted: {pred_deathdate}",
            "True OCCUPATION / Predicted OCCUPATION": f"True: {true_occupation} | Predicted: {pred_occupation}",
            "Token Evaluation": token_eval
        }
        rows.append(row)
    record_df = pd.DataFrame(rows)
    overall_list = []

    if overall_metrics is None:
        overall_metrics = exact_matching_accuracy_per_document(test_data, test_input_ids, predictions, tokenizer,
                                                               id2label)
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
    classes = ["NAME", "BIRTHDATE", "DEATHDATE", "OCCUPATION"]

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
