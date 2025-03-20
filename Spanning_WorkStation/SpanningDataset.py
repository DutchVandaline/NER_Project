import torch
from torch.utils.data import Dataset
import json


class SpanningDataset(Dataset):
    def __init__(self, input_ids, attention_masks, spans, entity_fields):
        """
        Args:
            input_ids (List[List[int]]): 각 문장의 토큰 아이디 리스트.
            attention_masks (List[List[int]]): 각 문장의 어텐션 마스크.
            spans (List): 각 문장의 스팬 정보 리스트.
                          예: [{'start': 3, 'end': 5, 'label': 'PER'}, ...] 또는 JSON 문자열.
            entity_fields (dict): 라벨을 정수 인덱스로 매핑하는 딕셔너리.
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.spans = spans
        self.entity_fields = entity_fields

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        spans_info = self.spans[idx]

        # spans_info가 문자열이면, JSON 형식으로 파싱
        if isinstance(spans_info, str):
            spans_info = json.loads(spans_info)

        # 기본 라벨은 'O'로 지정 (entity_fields에 'O'가 0 등으로 매핑되어 있다고 가정)
        default_label = self.entity_fields.get('O', 0)
        labels = [default_label] * len(tokens)

        # 각 스팬 정보를 토큰 단위의 라벨에 반영
        for span in spans_info:
            start = span['start']
            end = span['end']
            label_str = span['label']
            label_id = self.entity_fields.get(label_str, default_label)
            for i in range(start, min(end + 1, len(tokens))):
                labels[i] = label_id

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
