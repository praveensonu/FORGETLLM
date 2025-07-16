import torch
from typing import Dict, List, Any


def dpo_retain_collator(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]: # for batch dpo and npo which has factor like gradient difference
    """
    Collates samples from CombinedForgetRetainDataset.
    Each sample is a dict, potentially including:
    {
        'answer_input_ids': Tensor, 'answer_labels': Tensor, 'answer_attention_mask': Tensor,
        'idk_input_ids': Tensor, 'idk_labels': Tensor, 'idk_attention_mask': Tensor,
        'factor': float,
        'original_index': Tensor (scalar, long)
    }
    Returns a batch dict with stacked tensors. 'factor' is converted to a float tensor.
    """
    if not samples:
        return {}

    
    batch = {}
    first_sample_keys = samples[0].keys()

    for key in first_sample_keys:
        values = [sample[key] for sample in samples]

        if key == 'factor':
            batch[key] = torch.tensor(values, dtype=torch.float)
        elif isinstance(values[0], torch.Tensor):
            
            batch[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, bool, str)):

            batch[key] = values 
        else:
            batch[key] = values
            
    return batch