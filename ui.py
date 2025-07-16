import gradio as gr, tempfile
import shutil, tempfile, os
from gradio.components import File as GradioFile
from helpers.config import Config
from gradient_ascent.grad_ascent import run_gradient_ascent
from gradient_diff.grad_diff import run_grad_diff


# Map dropdown label ➜ callable
METHOD_REGISTRY = {
    "gradient_ascent": run_gradient_ascent,
    "gradient_diff": run_grad_diff,

}

def tmp_copy(upload: "GradioFile | str | bytes | None") -> str | None:
    """
    Accepts anything gr.File may return:
      • str path  (when type='filepath')
      • NamedString (default 'auto')
      • bytes      (when type='binary')
    and returns a stable temp-file path, or None.
    """
    if upload is None:
        return None

    if isinstance(upload, str):
        return upload


    if hasattr(upload, "name"):           
        src_path = upload.name             
        dst_path = tempfile.mkstemp(prefix="upl_")[1]
        shutil.copy(src_path, dst_path)
        return dst_path

    if isinstance(upload, (bytes, bytearray)):
        dst_path = tempfile.mkstemp(prefix="upl_")[1]
        with open(dst_path, "wb") as f:
            f.write(upload)
        return dst_path
    
    raise TypeError(f"Unsupported upload object: {type(upload)}")

def train_interface(method,
                    access_token, 
                    model_id,
                    lora_r, 
                    lora_alpha, 
                    lora_dropout,
                    learning_rate, 
                    batch_size,
                    grad_acc_steps, 
                    num_epochs, 
                    weight_decay,
                    forget_csv, 
                    retain_csv, 
                    gpu_ids):

    cfg = Config()

    # ---- user hyper-params ---------------------------------
    cfg.loss_type                  = method
    cfg.access_token               = access_token.strip()
    cfg.model_id                   = model_id
    cfg.LoRA_r                     = int(lora_r)
    cfg.LoRA_alpha                 = int(lora_alpha)
    cfg.LoRA_dropout               = float(lora_dropout)
    cfg.lr                         = float(learning_rate)
    cfg.batch_size                 = int(batch_size)
    cfg.gradient_accumulation_steps= int(grad_acc_steps)
    cfg.num_epochs                 = int(num_epochs)
    cfg.weight_decay               = float(weight_decay)
    cfg.gpu_ids                    = gpu_ids.strip()

    if cfg.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    # ---- uploaded data -------------------------------------
    cfg.forget_path   = tmp_copy(forget_csv)
    cfg.retain_path = tmp_copy(retain_csv)

    cfg.save_dir = cfg.save_dir

    # --- Minimal Validation -------------------------------
    if cfg.forget_path is None:
        return "Please upload a Forget CSV first!"
    
    if method in ('gradient_diff') and cfg.retain_path is None:
        return "This method requires a Retain file, please upload it."
    
    # ---- dispatch to the chosen method ---------------------
    runner = METHOD_REGISTRY.get(method)
    if runner is None:
        return f"Method '{method}' not implemented."
    
    try:
        log = runner(cfg)
    except Exception as e:
        log = f"Training failed:\n{e}"

    return log


# ---------------- Gradio layout ----------------------------
with gr.Blocks(title="Minimal LLM Un-learning Trainer") as demo:
    gr.Markdown("Minimal LLM Un-learning tool")

    gpu_ids_in = gr.Textbox(
        label = "GPU id(s) for CUDA_VISIBLE_DEVICES",
        value = "0",
        placeholder = 'e.g. 0 or 0,1,2,3 or leave empty to use all the devices.'
    )
    method_dd = gr.Dropdown(
        choices=list(METHOD_REGISTRY.keys()),
        value="gradient_ascent",
        label="Un-learning method",
    )

    with gr.Accordion("HuggingFace credentials", open=False):
        access_token_in = gr.Textbox(label="Token", type="password")
        model_id_in     = gr.Textbox(
            label="Model ID",
            value="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

    with gr.Row():
        lora_r_in       = gr.Number(label="LoRA r", value=8, precision=0)
        lora_alpha_in   = gr.Number(label="LoRA α", value=16, precision=0)
        lora_dropout_in = gr.Slider(label="Dropout", minimum=0, maximum=0.5,
                                    step=0.01, value=0.05)

    with gr.Row():
        lr_in        = gr.Number(label="Learning rate", value=2e-5)
        batch_in     = gr.Number(label="Batch size", value=4, precision=0)
        grad_in      = gr.Number(label="Grad-acc steps", value=1, precision=0)
        epochs_in    = gr.Number(label="Epochs", value=4, precision=0)
        wd_in        = gr.Number(label="Weight decay", value=0.01)

    gr.Markdown("Upload datasets (only what your method needs)")
    forget_up   = gr.File(label="Forget File",   file_types=[".csv"])
    retain_up = gr.File(label="Retain File", file_types=[".csv"])

    run_btn  = gr.Button("🚀 Train")
    console  = gr.Textbox(label="Console log", lines=18, interactive=False)

    run_btn.click(
        train_interface,
        inputs=[
            method_dd,
            access_token_in, model_id_in,
            lora_r_in, lora_alpha_in, lora_dropout_in,
            lr_in, batch_in, grad_in, epochs_in, wd_in,
            forget_up, retain_up,
            gpu_ids_in,
        ],
        outputs=console,
    )

demo.launch(share = True)
