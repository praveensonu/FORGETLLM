import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Tuple, Any

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer
    num_question_tokens = len(tokenizer.tokenize(question, add_special_tokens=False)) #this is important, we 
    encoded = tokenizer(
        full_text,
        add_special_tokens=False, #this is important, we keep false cause we already added the special tokens from template
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    #change label to -100 for question tokens, including assistant header and end of header.
    for i in range(num_question_tokens): label[i] = -100
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)



class VanillaDPODataset(Dataset):

    def __init__(self, forget_data: pd.DataFrame, tokenizer: Any,
                 max_length: int,
                 question_key: str = 'question',
                 answer_key: str = 'answer',
                 idk_key: str = 'idk'):
        if not all(k in forget_data.columns for k in [question_key, answer_key, idk_key]):
             raise ValueError(f"forget_data must contain columns: {question_key}, {answer_key}, {idk_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
        self.ik = idk_key

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        row = self.forget_data.iloc[idx]
        q = row[self.qk]
        ans = row[self.ak]
        idk = row[self.ik]

        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, ans,
                                                )
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, idk,
                                                )

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
        }


class ForgetIdkRetainDatasetRandom(Dataset):
    """
    For each row in forget_data, returns a dictionary containing three items:
    1. The forget question paired with its original answer.
    2. The forget question paired with its "I don't know" answer.
    3. A RANDOMLY selected question-answer pair from the retain_data.

    Output format is a dictionary of tensors:
      {
        'answer_input_ids': ..., 'answer_labels': ..., 'answer_attention_mask': ...,
        'idk_input_ids': ..., 'idk_labels': ..., 'idk_attention_mask': ...,
        'retain_input_ids': ..., 'retain_labels': ..., 'retain_attention_mask': ...,
      }
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        question_key: str = 'question',
        answer_key: str = 'answer',
        idk_key: str = 'idk',
    ):
        # validate
        if not all(col in forget_data.columns for col in [question_key, answer_key, idk_key]):
            raise ValueError(f"forget_data must contain: {question_key}, {answer_key}, {idk_key}")
        if not all(col in retain_data.columns for col in [question_key, answer_key]):
            raise ValueError(f"retain_data must contain: {question_key}, {answer_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk, self.ak, self.ik = question_key, answer_key, idk_key

    def __len__(self):
        # The length of an epoch is determined by the number of samples to forget.
        return len(self.forget_data)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        f_row = self.forget_data.iloc[idx]

        # CHANGED: The retain sample is chosen RANDOMLY from the entire retain set.
        random_retain_idx = torch.randint(0, len(self.retain_data), (1,)).item()
        r_row = self.retain_data.iloc[random_retain_idx]

        # --- The rest of the logic remains the same ---

        # Process forget sample with its original answer
        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, ans)

        # Process forget sample with its "idk" answer
        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, idk)

        # Process the RANDOMLY CHOSEN retain sample
        retain_q = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, retain_q, retain_ans)

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }