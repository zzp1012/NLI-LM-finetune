import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Callable

class Vectorizer():
    """Vectorizer for NLI task"""
    def __init__(self, tokenizer: Callable, max_seq_len: int):
        """Vectorizer for NLI task

        Args:
            tokenizer (Callable): tokenizer
            max_seq_len (int): max sequence length
        
        Returns:
            None
        """
        self.tokenizer = tokenizer
        self.__max_seq_len = max_seq_len
    
    def vectorize(self, text: str) -> np.ndarray:
        """Vectorize text

        Args:
            text (str): text

        Returns:
            np.ndarray: vectorized text
        """
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.encode(text)[1:-1]
        assert len(tokenized_text) == len(indexed_tokens)

        segments_ids, seg_flag = [], False
        for i in range(len(tokenized_text)):
            if tokenized_text[i] == "[SEP]" and seg_flag == False:
                seg_flag = True
            
            if seg_flag:
                segments_ids.append(1)
            else:
                segments_ids.append(0)
        
        attention_mask = [1] * len(indexed_tokens)

        # padding
        if len(indexed_tokens) < self.__max_seq_len:
            indexed_tokens += [0] * (self.__max_seq_len - len(indexed_tokens))
            segments_ids += [1] * (self.__max_seq_len - len(segments_ids))
            attention_mask += [0] * (self.__max_seq_len - len(attention_mask))

        return {
            "ids": np.array(indexed_tokens),
            "types": np.array(segments_ids),
            "masks": np.array(attention_mask)
        }


class NLIDataset(Dataset):
    """Dataset for NLI task"""
    def __init__(self, 
                 path: str, 
                 tokenizer: Callable,):
        """Dataset for NLI task

        Args:
            path (str): path to data
            tokenizer (Callable): tokenizer [bert-base-uncased]
        
        Returns:
            None
        """
        self.data_df = self.preprocess(self.load(path))
        self.tokenizer = tokenizer     
        
        self.vectorizer = Vectorizer(
            tokenizer, 
            self.__get_max_len(
                self.data_df['sentence'], tokenizer
            )
        )
    
    def __len__(self) -> int:
        """Get length of dataset

        Returns:
            int: length of dataset
        """
        return len(self.data_df)
    
    def __getitem__(self, index: int) -> dict:
        """Get item from dataset

        Args:
            index (int): index of item

        Returns:
            dict: item
        """
        row = self.data_df.iloc[index]
        return {
            "id": row['id'],
            **self.vectorizer.vectorize(row['sentence']),
            **({"label": row['label']} if 'label' in row else {})
        }
    
    def preprocess(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
            1. combine sentence1 and sentence2 with sep token
            2. add cls token at the beginning
            3. add eos token at the end
        
        Args:
            data_df (pd.DataFrame): data
    
        Returns:
            pd.DataFrame: preprocessed data
        """
        sentences = []
        for sent1, sent2 in zip(data_df['sentence1'], data_df['sentence2']):
            sentences.append(f"[CLS] {sent1} [SEP] {sent2} [SEP]")
        
        # add new column to data_df
        data_df['sentence'] = sentences
        return data_df
    
    @classmethod
    def load(cls, path: str,) -> pd.DataFrame:
        """Load data from path

        Args:
            path (str): path to data

        Returns:
            pd.DataFrame: data
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip().split("\t"))
        # read a tsv file
        data = pd.DataFrame(lines[1:], columns=lines[0])
        # change the label to 0, 1, 2
        if 'label' in data.columns:
            data['label'] = data['label'].map({
                'contradiction': 0,
                'entailment': 1, 
                'neutral': 2, 
            })
            # check the label
            assert data['label'].isin([0, 1, 2]).all()
        return data
    
    @classmethod
    def __get_max_len(cls, 
                      data: pd.Series, 
                      tokenizer: Callable) -> int:
        """Get max length of sequence

        Args:
            data_df (pd.DataFrame): data
            tokenizer (Callable): tokenizer

        Returns:
            int: max length of sequence
        """
        len_func = lambda text: len(tokenizer.tokenize(text))
        return data.map(len_func).max()

# test 
if __name__ == "__main__":
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = NLIDataset(
        path="../data/train.tsv",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    print(next(iter(dataloader)))
    