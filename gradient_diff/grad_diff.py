import os 
from helpers.config import Config


def run_grad_diff(cfg: Config) -> str:

    # -------- GPU selection ---------------------------------
    if cfg.gpu_ids:                       # "" means “use all GPUs”
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
        print("Using GPU(s):", cfg.gpu_ids)

    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from .data_module import DualDatasetRandom
    from .collators import custom_gd_collator_forget
    from .trainer import GradDiffTrainer
    from helpers.template import LLAMA3_CHAT_TEMPLATE

    accelerator = Accelerator()
    if cfg.forget_path is None:
        raise ValueError("cfg.forget_path is None – upload a Forget CSV first!")

    print("Loading Forget CSV:", cfg.forget_path)
    forget = pd.read_csv(cfg.forget_path)
    retain = pd.read_csv(cfg.retain_path)

    # ---- Model & tokenizer --------------------------------
    print(f"\nLoading tokenizer {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, token=cfg.access_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading model {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, token=cfg.access_token
    )

    # ---- LoRA ---------------------------------------------
    lora_cfg = LoraConfig(
        r              = cfg.LoRA_r,
        lora_alpha     = cfg.LoRA_alpha,
        lora_dropout   = cfg.LoRA_dropout,
        target_modules = cfg.LoRa_targets,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        modules_to_save=[]          # ← fixes the NoneType bug
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # ---- Prompt formatting --------------------------------
    def to_chat(df):
        df["question"] = df["question"].apply(
            lambda x: LLAMA3_CHAT_TEMPLATE.format(question=x)
        )
        df["answer"] = df["answer"] + tokenizer.eos_token
        return df

    forget = to_chat(forget)
    retain = to_chat(retain)

    # ---- Trainer stuff ------------------------------------
    training_args = TrainingArguments(
        output_dir                   = cfg.save_dir,
        overwrite_output_dir         = True,
        learning_rate                = cfg.lr,
        per_device_train_batch_size  = cfg.batch_size,
        num_train_epochs             = cfg.num_epochs,
        weight_decay                 = cfg.weight_decay,
        logging_dir                  = f"{cfg.save_dir}/logs",
        evaluation_strategy          = "no",
        label_names                  = ["labels"],
        bf16                         = True,
        gradient_accumulation_steps  = cfg.gradient_accumulation_steps,
    )

    dataset = DualDatasetRandom(
        forget_data = forget,
        retain_data = retain,
        tokenizer   = tokenizer,
        max_length  = cfg.max_length,
    )

    trainer = GradDiffTrainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
        tokenizer     = tokenizer,
        data_collator = custom_gd_collator_forget,
    )
    trainer.train()
    accelerator.wait_for_everyone()

    tokenizer.save_pretrained(cfg.save_dir)
    model.save_pretrained(cfg.save_dir)

    return f"✅ Training finished. Adapter & tokenizer saved to {cfg.save_dir}"


# ⬇ allow CLI use too --------------------------------------
if __name__ == "__main__":
    cfg = Config()
    cfg.forget_path = "./data/dpo_forget_idk.csv"   # fallback for CLI runs
    print(run_grad_diff(cfg))
