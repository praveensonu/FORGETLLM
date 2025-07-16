from torch.utils.data import Dataset
import torch
import pandas as pd

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



class DualDatasetRandom(Dataset):
    """
    TOFU way of implementation.

    Args:
        forget_data (pd.DataFrame): DataFrame for forgetting.
        retain_data (pd.DataFrame): DataFrame for retaining.
        tokenizer: tokenizer instance to process text.
        max_length (int): maximum sequence length.
        question_key (str): column name for questions.
        answer_key (str): column name for answers.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        forget_idx = idx
        # A new random sample is chosen every time __getitem__ is called.
        retain_idx = torch.randint(0, len(self.retain), (1,)).item()

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)