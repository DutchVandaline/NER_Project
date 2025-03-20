import os
import json
import pandas as pd
from transformers import AutoTokenizer

from Spanning_WorkStation.Result_Calculation import compute_span_metrics, exact_matching_accuracy_span, print_text_and_entity_predictions_span
from Spanning_WorkStation.Spanning_Labeling import labeling


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df_train = pd.read_csv(r"C:\junha\Datasets\WikiBio_wikipedia-biography-dataset\train.csv")
df_test = pd.read_csv(r"C:\junha\Datasets\WikiBio_wikipedia-biography-dataset\val.csv")
train_data = df_train.to_dict(orient="records")
test_data = df_test.to_dict(orient="records")
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_input_ids, train_attention_masks, train_spans = labeling(train_data, tokenizer)
test_input_ids, test_attention_masks, test_spans = labeling(test_data, tokenizer)
train_output_file = "../Results/Spanning/train_spanning_results.txt"
with open(train_output_file, "w", encoding="utf-8") as f:
    for spans in train_spans:
        f.write(json.dumps(spans, ensure_ascii=False) + "\n")
print("Train spanning results saved:", train_output_file)
test_output_file = "../Results/Spanning/test_spanning_results.txt"
with open(test_output_file, "w", encoding="utf-8") as f:
    for spans in test_spans:
        f.write(json.dumps(spans, ensure_ascii=False) + "\n")
print("Test spanning results saved:", test_output_file)
# 평가 함수 호출 시 반드시 두 개의 인자를 전달해야 합니다.
span_metrics = compute_span_metrics(test_data, test_spans)
exact_match_metrics = exact_matching_accuracy_span(test_data, test_spans)
print_text_and_entity_predictions_span(test_data, test_spans)

