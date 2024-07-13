from typing import List, Dict, Callable

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, TensorDataset

TORCH_IGNORE_INDEX = -1
LIMIT_N_SAMPLES = None  # can set a number for debugging purposes

"""
Loading the ListOps dataset and the OpenWebText dataset.
>> Referring to LRA's paper and loosely based on: https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/text_classification/input_pipeline.py 
"""


def preprocess_lra(s: str) -> List[str]:
    # LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
    # non-alphanumeric characters.
    # https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()


def preprocess_text(s: str) -> List[str]:
    return list(s.strip().lower())  # tokenize as chars


class Tokenizer:
    # Similar to https://github.com/state-spaces/s4/blob/main/src/dataloaders/lra.py#L306
    def __init__(self, max_length=1024):
        self.token_to_token_id: dict = {
            "<pad>": 0,
            "<unk>": 1,
            "<eos>": 2,
            # "<bos>": 3,  #[DISABLED]
        }
        self.tokenizer: callable = None
        self.max_length = max_length  # TODO parameter  # max sequence length, for padding and truncation

    def __call__(self, s: str, preprocess_fn: Callable[[str], List[str]] = None,
                 pre_padding: bool = True) -> (List[int], List[int]):
        """
        Encoding a string into a list of integers, and applying padding if needed.
        :param s: string to tokenize
        :param preprocess_fn: function to apply prior to tokenization
        :param pre_padding: whether to apply padding from left (True) or not at to pad at all (False).
        :return: the tokenized string as a list of integers, and the pad mask (can also serve as attention mask)
        """
        if preprocess_fn:
            s = preprocess_fn(s)
        out = [
                  self.token_to_token_id.get(token, self.token_to_token_id["<unk>"]) for token in s
              ][:(self.max_length - 1)]
        out += [self.token_to_token_id['<eos>']]
        pad_mask = [1] * len(out)  # 1 for tokens, 0 for padding
        if pre_padding:
            # add pre-padding (<pad> token until max_length)
            out = [self.token_to_token_id['<pad>']] * (self.max_length - len(out)) + out
            pad_mask = [0] * (self.max_length - len(pad_mask)) + pad_mask

        return out, pad_mask  # pad_mask can also be used as attention mask

    def load(self):
        pass  # TODO load from file

    def save(self):
        pass  # TODO

    def add_tokens_from(self, samples: List[str], preprocess_fn: Callable[[str], List[str]] = None):
        print(f"Adding tokens from {len(samples)} samples")
        for sample in samples:
            if preprocess_fn is not None:
                sample = preprocess_fn(sample)
            for token in sample:  # each sample is a list of strings (=token)
                if token not in self.token_to_token_id:  # word has not been assigned an index yet
                    self.token_to_token_id[token] = len(self.token_to_token_id)  # Assign each word with a unique index
        print(f"Updated vocabulary size to {len(self.token_to_token_id)}")

    @property
    def vocab_size(self):
        return len(self.token_to_token_id)


class ListOpsDataset(TensorDataset):
    # https://github.com/NYU-MLL/spinn/tree/listops-release
    def __init__(self,
                 split: str = 'train',
                 tokenizer=None,  # TODO support importing tokenizer
                 n_samples: int = LIMIT_N_SAMPLES,
                 task: str = 'classification',
                 max_length: int = 1024,
                 ):
        """
        :param split: what split to load ('train' | 'test')
        :param tokenizer: tokenizer to use (none means a new one will be created)
        :param n_samples: optional number of samples to load (mainly for testing)
        :param is_auto_regressive_task: whether to make the task auto-regressive
        """
        super(ListOpsDataset, self).__init__()

        assert task in ['classification', 'auto_regressive']
        self.task = task  # can also be modified post-init

        # load:
        data = pd.read_csv(f'./data/listops/{split}_d20s.tsv', sep='\t',
                           header=None, names=['target', 'source'])
        data["source"] = data["source"].apply(str)
        data["target"] = data["target"].apply(int)
        if n_samples is not None:  # trim dataset (mainly to testing)
            data = data.iloc[:n_samples]
        self.data = data
        self.n_classes = len(data['target'].unique())

        # initialize custom tokenizer:
        if tokenizer is None:
            self.tokenizer = Tokenizer(max_length)
            self.tokenizer.add_tokens_from(data['source'].tolist(), preprocess_fn=preprocess_lra)
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """:returns the tokenized source, padding/attention mask, and the target class to predict"""
        # tokenize the sample:
        source = self.data.iloc[idx, 1]
        source, pad_mask = self.tokenizer(source, preprocess_fn=preprocess_lra)

        # Define the target wrp to the task:
        if self.task == 'classification':
            # for classification, the target is a single integer
            target = self.data.iloc[idx, 0]
        elif self.task == 'auto_regressive':
            # for auto-regressive, the target is a list of integers
            source, target = source[:-1], source[1:]
            pad_mask = pad_mask[:-1]

            # Ignore the padding tokens (`-1` is an ignored index in torch's CrossEntropyLoss)
            # * Note: this means that the model will not be penalized on predictions after a padding token
            target = [t if m == 1 else TORCH_IGNORE_INDEX for t, m in zip(target, pad_mask)]

        else:
            raise ValueError(f"Unknown task: {self.task}")

        source, pad_mask, target = torch.tensor(source), torch.tensor(pad_mask), torch.tensor(target)
        return source, pad_mask, target


class OpenWebTextDataset(TensorDataset):
    def __init__(self, split='train', tokenizer=None, seq_len=1024,
                 n_samples: int = LIMIT_N_SAMPLES):
        super().__init__()
        self.seq_len = seq_len

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

        self.dataset = dataset[split]['text']
        self.dataset = [s for s in self.dataset if len(s) > 0]  # filter empty strings

        if tokenizer is None:
            self.tokenizer = Tokenizer(max_length=seq_len)
            self.tokenizer.add_tokens_from(self.dataset,
                                           preprocess_fn=preprocess_text)  # TODO isn't it just a large tet sample?

        # TODO make sure this can tokenize the ListOps dataset
        # TODO ponder about this tokenization - is it correct ot use this character-level tokenization? should we do something else to support ListOps?

        # Tokenize all the dataset with it, and concat all the tokens
        self.dataset = [self.tokenizer(s, preprocess_fn=preprocess_text, pre_padding=False)[0] for s in self.dataset]
        # TODO note that this means we do not pretrain with the padding token.. maybe it's not good? (seems that Ankit didn't pad in pretrain)
        self.dataset = [token for sample in self.dataset for token in sample]

        if n_samples is not None:  # just for testing
            self.dataset = self.dataset[:n_samples]

    def __len__(self):
        return len(self.dataset) - self.seq_len

    def __getitem__(self, idx):
        """:returns a sequence of `seq_len` tokens, padding/attention mask, and the target token to predict"""
        sample = self.dataset[idx: idx + self.seq_len]
        sample, target = sample[:-1], sample[1:]  # auto-regressive task
        return torch.tensor(sample), torch.ones(self.seq_len - 1), torch.tensor(target)
