from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    # ────────── experiment / framework ──────────
    loss_type: str = "gradient_ascent"  # dropdown in UI
    access_token: str = ""              # HuggingFace token

    # ────────── model backbone ──────────
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    beta : float = 0.1
    gamma : float = 1.0
    alpha : float = 1.0

    # ────────── LoRA hyper-params ──────────
    LoRA_r: int = 8
    LoRA_alpha: int = 16
    LoRA_dropout: float = 0.05
    LoRa_targets: list[str] = field(default_factory=lambda: [
        "v_proj", "k_proj", "up_proj",
        "o_proj", "gate_proj", "q_proj", "down_proj"
    ])

    # ────────── optimiser / training ──────────
    lr: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_epochs: int = 4
    weight_decay: float = 0.01
    max_length: int = 256  # sequence length for tokenisation

    # ────────── compute resource ──────────
    gpu_ids: str = "0"  # UI textbox ➜ CUDA_VISIBLE_DEVICES; "" = all GPUs

    # ────────── data paths (injected by UI uploads) ──────────
    forget_path: str | None = None      # mandatory for gradient_ascent
    retain_path: str | None = None    # e.g. for DPO/NPO

    # ────────── output ──────────
    save_dir: str = field(init=False)   # set in __post_init__

    # ────────── helpers ──────────
    def __post_init__(self) -> None:
        # derive exp-specific save directory
        self.save_dir = f"./outputs/{self.loss_type}_model"

    # if you ever change loss_type after construction,
    # call cfg.refresh_paths() to keep save_dir in sync.
    def refresh_paths(self) -> None:
        self.save_dir = f"./outputs/{self.loss_type}_model"