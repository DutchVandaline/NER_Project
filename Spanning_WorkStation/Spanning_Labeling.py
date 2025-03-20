import re

entity_fields = {
    "NAME": "name",
    "BIRTH_DATE": "birth_date",
    "DEATH_DATE": "death_date",
    "OCCUPATION": "occupation",
}

def create_spanning_annotations_with_offsets(text, tokens, offsets, record):
    spans = {}
    for tag, field in entity_fields.items():
        entity_value = record.get(field)
        try:
            entity_value = str(entity_value).strip() if entity_value is not None else ""
        except Exception:
            entity_value = ""
        if entity_value:
            pattern = re.escape(entity_value)
            match = re.search(pattern, text)
            if match:
                ent_start, ent_end = match.span()
                start_idx = None
                end_idx = None
                for i, (token_start, token_end) in enumerate(offsets):
                    if token_start == token_end == 0:
                        continue
                    if start_idx is None and token_end > ent_start:
                        start_idx = i
                    if token_start < ent_end:
                        end_idx = i
                if start_idx is not None and end_idx is not None:
                    spans[tag] = {"start": start_idx, "end": end_idx, "text": entity_value}
    return spans

def labeling(data_list, tokenizer):
    all_input_ids = []
    all_attention_masks = []
    all_spans = []
    for record in data_list:
        text = str(record.get("doc", ""))
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
        spans = create_spanning_annotations_with_offsets(text, tokens, offsets, record)
        all_input_ids.append(input_ids_list)
        all_attention_masks.append(attention_mask_list)
        all_spans.append(spans)
    return all_input_ids, all_attention_masks, all_spans
