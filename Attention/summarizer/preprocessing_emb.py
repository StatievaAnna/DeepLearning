import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from collections import namedtuple

DEVICE = torch.device('cuda')

_tokenizer = None
_model = None

def get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_name = "cointegrated/rubert-tiny2"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(DEVICE)
        _model.eval()
    return _tokenizer, _model

class TextPairDataset(Dataset):
    def __init__(self, data):
        self.source_texts = data['text'].tolist()
        self.target_texts = data['title'].tolist()

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        return self.source_texts[idx], self.target_texts[idx]

# def collate_fn(batch):
#     source_texts, target_texts = zip(*batch)
#     return list(source_texts), list(target_texts)

def collate_fn(batch):
    # Создаем класс для хранения батча с именованными полями
    class Batch:
        def __init__(self, source, target):
            self.source = list(source)
            self.target = list(target)
    
    # Распаковываем batch (каждый элемент - кортеж (source_text, target_text))
    source_texts, target_texts = zip(*batch)
    
    # Возвращаем объект Batch
    return Batch(source_texts, target_texts)

def subsequent_mask(size):
    mask = torch.ones(size, size, device=DEVICE).triu_()
    return mask.unsqueeze(0) == 0

def preprocessing(data, batch_size=16):
    """
    Возвращает итераторы (DataLoader), которые возвращают списки строк
    для подачи в контекстный эмбеддер (rubert-tiny2 и т.п.)
    """
    tokenizer, model = get_model_and_tokenizer()

    train_data, test_data = train_test_split(data, test_size=0.15, shuffle=True)

    train_dataset = TextPairDataset(train_data.reset_index(drop=True))
    test_dataset = TextPairDataset(test_data.reset_index(drop=True))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, collate_fn=collate_fn)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

    d_model = model.config.hidden_size

    return train_loader, test_loader, tokenizer, d_model

def get_embeddings_and_masks(texts):
    # texts — список строк (батч)
    tokenizer, model = get_model_and_tokenizer()
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # print('что-то закодированное', )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        embeddings = outputs.last_hidden_state

    return embeddings, attention_mask

# def convert_batch_with_pretrain_emb(batch):
#     source_texts = batch.source  # список строк
#     target_texts = batch.target

#     source_embeddings, source_mask = get_embeddings_and_masks(source_texts)
#     target_embeddings, target_mask = get_embeddings_and_masks(target_texts)

#     # Преобразуем attention_mask в формат, подходящий для масок в Transformer
#     source_mask = source_mask.unsqueeze(1).unsqueeze(2)
#     target_mask = target_mask.unsqueeze(1).unsqueeze(2)

#     return source_embeddings, target_embeddings, source_mask, target_mask

def convert_batch_with_pretrain_emb(batch):

    source_texts = batch.source  # список строк
    target_texts = batch.target

    source_embeddings, source_mask = get_embeddings_and_masks(source_texts)
    target_embeddings, target_mask = get_embeddings_and_masks(target_texts)

    # # Печатаем размерности эмбеддингов и масок
    # print(f"source_embeddings shape: {source_embeddings.shape}")  # [batch_size, seq_len, emb_dim]
    # print(f"source_mask shape: {source_mask.shape}")              # [batch_size, seq_len]
    # print(f"target_embeddings shape: {target_embeddings.shape}")
    # print(f"target_mask shape: {target_mask.shape}")

    # Преобразуем attention_mask в формат, подходящий для масок в Transformer
    # source_mask = source_mask.unsqueeze(1).unsqueeze(2)
    target_mask = target_mask.unsqueeze(-2) & subsequent_mask(target_embeddings.size(-2)).type_as(target_mask)

    source_mask = source_mask.unsqueeze(1)
    # Печатаем размерности после преобразования масок
    # print(f"Transformed source_mask shape: {source_mask.shape}")  # [batch_size, 1, 1, seq_len]
    # print(f"Transformed target_mask shape: {target_mask.shape}")

    return source_embeddings, target_embeddings, source_mask, target_mask

# def make_mask(source_inputs, target_inputs, pad_idx):
#     source_mask = (source_inputs != pad_idx).unsqueeze(-2)
#     target_mask = (target_inputs != pad_idx).unsqueeze(-2)
#     target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
#     return source_mask, target_mask

# def convert_batch_with_pretrain_emb(batch, pad_idx=1):

#     source_texts = batch.source  # список строк
#     target_texts = batch.target

#     # Печатаем количество элементов в батче (сколько примеров)
#     print(f"Batch size (number of examples): source={len(source_texts)}, target={len(target_texts)}")

#     source_inputs, _ = get_embeddings_and_masks(source_texts)
#     target_inputs, _ = get_embeddings_and_masks(target_texts)

#     source_inputs, target_inputs = source_inputs.transpose(0, 1), target_inputs.transpose(0, 1)
#     source_inputs, target_inputs = source_inputs.to(DEVICE).transpose(0, 1), target_inputs.to(DEVICE).transpose(0, 1)
#     source_inputs, target_inputs = source_inputs.to(DEVICE), target_inputs.to(DEVICE)
#     source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)

#     print(f'размеры source_mask, target_mask: {source_mask.shape, target_mask.shape}')

#     return source_inputs, target_inputs, source_mask, target_mask