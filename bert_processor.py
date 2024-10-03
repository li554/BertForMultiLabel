import os

import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from preprocessor import EnglishPreProcessor
import pickle


class BertProcessor:
    def __init__(self, vocab_path, do_lower_case, max_seq_length) -> None:
        self.tokenizer = BertTokenizer(vocab_path, do_lower_case)
        self.max_seq_length = max_seq_length

    def encode(self, sentences):
        input_ids = []
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:self.max_seq_length - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        return input_ids

    def get_labels(self):
        return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def get_input_ids(self, x):
        input_ids = self.tokenizer.encode(x)
        if len(input_ids) > self.max_seq_length:
            input_ids = [input_ids[0]] + input_ids[1:self.max_seq_length - 1] + [input_ids[-1]]
        else:
            input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))
        return input_ids

    def get_label_ids(self, x):
        labels = []
        for item in x:
            labels.append(item if int(item) > 0 else -item)
        return labels

    def read_dataset(self, file_path):
        pickle_path = file_path.replace("csv", "pkl")
        if os.path.exists(pickle_path):
            file = open(pickle_path, mode='rb')
            dataset = pickle.load(file)
            file.close()
        else:
            data = pd.read_csv(file_path)
            data = data.fillna(0)
            # 对每一个comment_text做encode操作
            preprocessor = EnglishPreProcessor()
            data['label_ids'] = data.iloc[:, 2:].apply(self.get_label_ids, axis=1)
            label_ids = torch.tensor(list(data['label_ids'].values))
            # 英文预处理，包括去除停用词，大小写转换，删除无关字符，拆解单词等等
            tqdm.pandas(desc="english preprocess")
            data['comment_text'] = data['comment_text'].progress_apply(preprocessor)
            tqdm.pandas(desc="convert tokens to ids")
            data['input_ids'] = data['comment_text'].progress_apply(self.get_input_ids)
            input_ids = torch.tensor(list(data['input_ids'].values), dtype=torch.int)
            input_mask = torch.ones(size=(len(data), self.max_seq_length), dtype=torch.int)
            segment_ids = torch.zeros(size=(len(data), self.max_seq_length), dtype=torch.int)
            dataset = Data.TensorDataset(input_ids, input_mask, segment_ids, label_ids)
            file = open(pickle_path, mode='wb')
            pickle.dump(dataset, file)
            file.close()
        return dataset

    def train_val_split(self, dataset, batch_size, validation_split=0.2):
        shuffle_dataset = True
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = Data.SubsetRandomSampler(train_indices)
        valid_sampler = Data.SubsetRandomSampler(val_indices)
        train_iter = Data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        valid_iter = Data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
        return train_iter, valid_iter
