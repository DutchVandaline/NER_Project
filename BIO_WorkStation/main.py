import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, BertForTokenClassification, Trainer, TrainingArguments

from NERDataset import NERDataset
from NER_Labeling import labeling, id2label, label2id
from Result_Calculation import compute_metrics, compute_macro_metrics_per_document, exact_matching_accuracy_per_document, save_all_evaluation_excel, save_confusion_matrix_images, print_text_and_entity_predictions

df_train = pd.read_csv(r"/home/junha/WikiBIO/Dataset/train.csv")
df_test = pd.read_csv(r"/home/junha/WikiBIO/Dataset/val.csv")

train_data = df_train.to_dict(orient='records')
test_data = df_test.to_dict(orient='records')

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_input_ids, train_attention_masks, train_label_ids, train_sentences_iob = labeling(train_data, tokenizer)
test_input_ids, test_attention_masks, test_label_ids, test_sentences_iob = labeling(test_data, tokenizer)

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
