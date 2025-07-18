import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from accelerate import Accelerator
from transformers import Trainer


accelerator = Accelerator()

def get_batch_loss(output, labels):
    # when passed a ModelOutput or tuple, extract the first item
    if not torch.is_tensor(output):
        if hasattr(output, "logits"):
            output = output.logits
        else:
            output = output[0]

    shifted_labels = labels[..., 1:].contiguous()
    output         = output[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss    = loss_fn(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def compute_npo_loss(model, ref_model, win_inputs = None, lose_inputs = None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs cannot be None")
    
    win_log_ratio, lose_log_ratio = 0.0, 0.0

    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_outputs = model(**win_inputs)
        win_logits = win_outputs.logits
        win_loss = get_batch_loss(win_logits, win_inputs['labels'])
        with torch.no_grad():
            win_ref_outputs = ref_model(**win_inputs)
        win_ref_logits = win_ref_outputs.logits
        win_ref_loss = get_batch_loss(win_ref_logits, win_inputs['labels'])
        win_log_ratio = - (win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_outputs = model(**lose_inputs)
        lose_logits = lose_outputs.logits
        lose_loss = get_batch_loss(lose_logits, lose_inputs['labels'])
        with torch.no_grad():
            lose_ref_outputs = ref_model(**lose_inputs)
        lose_ref_logits = lose_ref_outputs.logits
        lose_ref_loss = get_batch_loss(lose_ref_logits, lose_inputs['labels'])
        lose_log_ratio = - (lose_loss - lose_ref_loss)

    loss =  -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_retain_loss(model, retain_inputs):  
    retain_outputs = model(**retain_inputs)
    retain_loss = 0.0
    retain_loss += retain_outputs.loss
    return retain_loss


class VanillaNPOTrainer(Trainer):
    def __init__(self,
                 ref_model,         
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 **hf_trainer_kwargs 
                ):
        super().__init__(**hf_trainer_kwargs)

        self.beta  = beta
        self.gamma = gamma
        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)
        

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids":      inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels":         inputs["answer_labels"],
        }
        idk_inputs = {
            "input_ids":      inputs["idk_input_ids"],
            "attention_mask": inputs["idk_attention_mask"],
            "labels":         inputs["idk_labels"],
        }

        forget_loss, forget_outputs = compute_npo_loss(
            model      = model,
            ref_model  = self.ref_model,
            win_inputs = None,
            lose_inputs=forget_inputs,
            beta       = self.beta,
        )
        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss


class RetainNPOTrainer(Trainer):
    def __init__(self,
                 ref_model,        
                 beta: float = 0.1,
                 gamma: float = 1.0,
                 alpha: float = 1.0,
                 **hf_trainer_kwargs 
                ):
        super().__init__(**hf_trainer_kwargs)

        self.beta  = beta
        self.gamma = gamma
        self.alpha = alpha

        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)
        

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids":      inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels":         inputs["answer_labels"],
        }
        idk_inputs = {
            "input_ids":      inputs["idk_input_ids"],
            "attention_mask": inputs["idk_attention_mask"],
            "labels":         inputs["idk_labels"],
        }
        retain_inputs = {
            "input_ids":      inputs["retain_input_ids"],
            "attention_mask": inputs["retain_attention_mask"],
            "labels":         inputs["retain_labels"],
        }

        forget_loss, forget_outputs = compute_npo_loss(
            model      = model,
            ref_model  = self.ref_model,
            win_inputs = None,
            lose_inputs=forget_inputs,
            beta       = self.beta,
        )

        retain_loss = compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss