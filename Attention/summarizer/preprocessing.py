# !shuf -n 10 news.csv
import random
import torch
import torch.nn as nn
import fasttext.util, fasttext
import numpy as np
from tqdm.auto import tqdm
from natasha import Segmenter, Doc
from torchtext.data import Field, Example, Dataset, BucketIterator
import pandas as pd
from sacremoses import MosesTokenizer

DEVICE = torch.device('cuda')

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'

def natasha_tokenize(text: str):
    segmenter = Segmenter()
    doc = Doc(text)
    doc.segment(segmenter)        # разбиваем на токены
    tokens = []
    tokens = [t.text.lower() for t in doc.tokens]
    return tokens

def get_word_field(tokenize):

    word_field = Field(
        tokenize=tokenize,      # наша функция
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        lower=True,                    # мы уже приводим к lower()
        batch_first=True
    )
    fields = [('source', word_field), ('target', word_field)]
    
    return word_field, fields

def create_dataset(data, tokenize):

    word_field, fields = get_word_field(tokenize)

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))

    dataset = Dataset(examples, fields)

    return dataset, word_field

def get_iter(train_dataset, test_dataset):


    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False
    )  
    return train_iter, test_iter

def prepare_word_field(word_field, dataset):
    fasttext.util.download_model('ru', if_exists='ignore')   # выдаст cc.ru.300.bin
    ft_model = fasttext.load_model('D:/05_Attention/05_Attention/seminar/cc.ru.300.bin')
    FT_DIM = ft_model.get_dimension() 

    word_field.build_vocab(dataset, min_freq=7)
    vocab_size = len(word_field.vocab)

    vectors = np.zeros((vocab_size, FT_DIM), dtype='float32')

    unk_vector = np.random.normal(scale=0.6, size=(FT_DIM,)) 

    # Инициализируем прогресс-бар
    progress_bar = tqdm(
        enumerate(word_field.vocab.itos), 
        total=len(word_field.vocab.itos),
        desc="Загрузка векторов"
    )

    for i, token in progress_bar:
        if token in (BOS_TOKEN, EOS_TOKEN):
            continue
        vectors[i] = ft_model.get_word_vector(token) if token in ft_model else unk_vector
        # Обновляем описание прогресс-бара текущим токеном (опционально)
        progress_bar.set_postfix({"Текущий токен": token[:10] + "..."})

    vectors[word_field.vocab.stoi[PAD_TOKEN]] = 0.0

    word_field.vocab.set_vectors(stoi=word_field.vocab.stoi,
                                vectors=torch.tensor(vectors),
                                dim=FT_DIM)
    return word_field

def get_embedding_layer(word_field):
    embedding_layer = nn.Embedding.from_pretrained(
        embeddings=word_field.vocab.vectors,
        freeze=False,
        padding_idx=word_field.vocab.stoi[PAD_TOKEN]
    )
    return embedding_layer

def preprocessing(data, tokenize=None):
    if tokenize == 'natasha':
        tokenize = natasha_tokenize
    elif tokenize == 'moses':
        tokenizer = MosesTokenizer(lang='ru')
        tokenize = tokenizer.tokenize

    dataset, word_field = create_dataset(data, tokenize)
    train_dataset, test_dataset = dataset.split(split_ratio=0.85)
    train_iter, test_iter = get_iter(train_dataset, test_dataset)
    word_field = prepare_word_field(word_field, dataset)

    if tokenize != None:
        embedding_layer = get_embedding_layer(word_field)
    else:
        embedding_layer = None 

    return train_iter, test_iter, word_field, embedding_layer

def subsequent_mask(size):
    mask = torch.ones(size, size, device=DEVICE).triu_()
    return mask.unsqueeze(0) == 0

def make_mask(source_inputs, target_inputs, pad_idx):
    source_mask = (source_inputs != pad_idx).unsqueeze(-2) # [bs, 1, sec_len]
    target_mask = (target_inputs != pad_idx).unsqueeze(-2) # [bs, 1, sec_len] * [sec_len, sec_len] = [bs, sec_len, sec_len]
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
    return source_mask, target_mask

def convert_batch(batch, pad_idx=1):
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_inputs, target_inputs = batch.source.to(DEVICE).transpose(0, 1), batch.target.to(DEVICE).transpose(0, 1)
    source_inputs, target_inputs = batch.source.to(DEVICE), batch.target.to(DEVICE)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)

    # print(f'размеры source_mask, target_mask: {source_mask.shape, target_mask.shape}')

    return source_inputs, target_inputs, source_mask, target_mask
