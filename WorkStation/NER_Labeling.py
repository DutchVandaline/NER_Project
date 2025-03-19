import re
import string



label2id = {
    "O": 0,
    "B-NAME": 1, "I-NAME": 2,
    "B-BIRTHDATE": 3, "I-BIRTHDATE": 4,
    "B-DEATHDATE": 5, "I-DEATHDATE": 6,
    "B-OCCUPATION": 7, "I-OCCUPATION": 8,


}
id2label = {v: k for k, v in label2id.items()}


def create_bio_labels_with_offsets(text, tokens, offsets, record):
    """
    text   : 원본 문장 (예: 질문수정 컬럼)
    tokens : 토크나이저로 분할된 토큰 리스트
    offsets: 각 토큰의 (start, end) 오프셋 정보
    record : 현재 데이터 레코드 (예: {"name": "...", "eatType": "...", ...})

    반환값: tokens와 길이가 같은 BIO 라벨 리스트
    """
    labels = ["O"] * len(tokens)

    entity_fields = {
        "NAME": "name",
        "BIRTHDATE": "birth_date",
        "DEATHDATE": "death_date",
        "OCCUPATION": "occupation"
    }

    for tag, field in entity_fields.items():
        try:
            entity_value = str(record.get(field)).strip() if record.get(field) is not None else ""
        except Exception:
            entity_value = ""
        # 엔티티 값이 존재하면 원본 텍스트 내에서 찾음
        if entity_value:
            # 특수 기호(예: "£")를 그대로 유지하면서 re.escape로 이스케이프 처리
            pattern = re.escape(entity_value)
            matches = re.finditer(pattern, text)
            for match in matches:
                ent_start, ent_end = match.span()
                b_assigned = False
                for i, (token_start, token_end) in enumerate(offsets):
                    # 오프셋 값이 [0,0]인 경우(예: 패딩)는 무시
                    if token_start == token_end == 0:
                        continue
                    # ±1 마진을 두어 오프셋 불일치를 보완 (특수 기호 등)
                    if token_end > ent_start and token_start < ent_end :
                        if not b_assigned:
                            labels[i] = f"B-{tag}"
                            b_assigned = True
                        else:
                            labels[i] = f"I-{tag}"
                # 첫번째 매칭만 사용 (여러 발생 시 추가 처리는 필요에 따라 수정)
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
            # 서브워드인 경우 앞 토큰과 붙여서
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


def labeling(data_list, tokenizer):
    all_input_ids = []
    all_attention_masks = []
    all_label_ids = []
    sentences_iob = []
    for record in data_list:
        text =  str(record.get("doc", ""))
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
