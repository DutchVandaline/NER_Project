import pandas as pd

def get_str(record, field):
    val = record.get(field)
    if pd.isnull(val):
        return ""
    else:
        return str(val).strip()

def compute_span_metrics(true_spans_list, pred_spans_list,
                         entity_types=["NAME", "BIRTHDATE", "DEATHDATE", "OCCUPATION"]):
    """
    각 문서에 대해 스패닝 결과(ground truth, prediction)를 받아 엔티티별 정밀도, 재현율, F1을 계산합니다.
    스패닝은 각 엔티티에 대해 {"start": ..., "end": ..., "text": ...} 형태로 제공됩니다.
    """
    metrics = {entity: {"TP": 0, "FP": 0, "FN": 0} for entity in entity_types}

    for true_spans, pred_spans in zip(true_spans_list, pred_spans_list):
        for entity in entity_types:
            true_entity = true_spans.get(entity, {})
            pred_entity = pred_spans.get(entity, {})
            true_text = true_entity.get("text", "").strip()
            pred_text = pred_entity.get("text", "").strip()

            # 둘 다 값이 있을 때 (엔티티가 존재하는 경우)
            if true_text and pred_text:
                # 텍스트가 정확히 일치하면 True Positive
                if true_text.lower() == pred_text.lower():
                    metrics[entity]["TP"] += 1
                else:
                    metrics[entity]["FP"] += 1
                    metrics[entity]["FN"] += 1
            # true에는 있는데 예측에 없으면 FN
            elif true_text and not pred_text:
                metrics[entity]["FN"] += 1
            # 예측에는 있는데 true에 없으면 FP
            elif not true_text and pred_text:
                metrics[entity]["FP"] += 1

    results = {}
    for entity in entity_types:
        TP = metrics[entity]["TP"]
        FP = metrics[entity]["FP"]
        FN = metrics[entity]["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results[entity] = {"precision": precision, "recall": recall, "f1": f1}
        print(f"Entity: {entity} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return results


def exact_matching_accuracy_span(test_data, true_spans_list, pred_spans_list,
                                 entity_types=["NAME", "BIRTHDATE", "DEATHDATE", "OCCUPATION"]):
    """
    각 문서별로 스패닝 결과가 정확하게 일치하는지를 평가합니다.
    test_data에는 원본 레코드가 들어있고, 여기서 실제 엔티티 값(예: name, birth_date 등)을 get_str()로 추출합니다.
    """
    overall_metrics = {entity: {"correct": 0, "total": len(test_data), "accuracy": 0.0}
                       for entity in entity_types}
    for idx, record in enumerate(test_data):
        for entity, field in zip(entity_types, ["name", "birth_date", "death_date", "occupation"]):
            true_entity = get_str(record, field).strip()
            pred_entity = pred_spans_list[idx].get(entity, {}).get("text", "").strip()
            if true_entity.lower() == pred_entity.lower():
                overall_metrics[entity]["correct"] += 1
    print("\nEntity-level Exact Matching Accuracy (Document-level):")
    for entity in entity_types:
        overall_metrics[entity]["accuracy"] = overall_metrics[entity]["correct"] / overall_metrics[entity]["total"]
        print(f"{entity} -> Accuracy: {overall_metrics[entity]['accuracy']:.4f}")
    return overall_metrics


def print_text_and_entity_predictions_span(test_data, true_spans_list, pred_spans_list):
    """
    각 문서별로 원본 텍스트와 함께, true 스패닝과 예측된 스패닝의 엔티티 값을 출력합니다.
    """
    print("Document-wise Entity Prediction Results:")
    for idx, record in enumerate(test_data):
        # 텍스트는 평가 데이터의 특정 필드("질문수정")에서 가져옵니다.
        text = get_str(record, "질문수정")
        true_entities = {
            "NAME": get_str(record, "name"),
            "BIRTHDATE": get_str(record, "birth_date"),
            "DEATHDATE": get_str(record, "death_date"),
            "OCCUPATION": get_str(record, "occupation"),
        }
        pred_entities = {
            "NAME": pred_spans_list[idx].get("NAME", {}).get("text", "").strip(),
            "BIRTHDATE": pred_spans_list[idx].get("BIRTHDATE", {}).get("text", "").strip(),
            "DEATHDATE": pred_spans_list[idx].get("DEATHDATE", {}).get("text", "").strip(),
            "OCCUPATION": pred_spans_list[idx].get("OCCUPATION", {}).get("text", "").strip(),
        }
        print(f"Text: {text}")
        for entity in true_entities:
            print(f"True {entity}: {true_entities[entity]} | Predicted {entity}: {pred_entities[entity]}")
        print("-" * 50)
