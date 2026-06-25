from __future__ import annotations

from transformers import TrainingArguments


def build_training_arguments(
    output_dir: str = "./afrivoices_checkpoints",
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    num_train_epochs: float = 10.0,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,

        # --- CPU-only enforcement (explicit, not assumed) --- #
        use_cpu=True,
        fp16=False,
        dataloader_pin_memory=False,  # pin_memory is a CUDA transfer optimization; irrelevant and wasteful on CPU.

        # --- Memory-safety: small real batch, simulated via accumulation --- #
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,

        # --- Optimization schedule --- #
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,  # Gradual warmup helps stabilize early CTC training, which can be loss-spiky at high LR.
        lr_scheduler_type="linear",
        weight_decay=0.005,  # Light regularization - low-resource multilingual data is prone to overfitting.

        # --- Evaluation cadence --- #
        eval_strategy="steps",
        eval_steps=500,

        # --- Disk I/O minimization (deliberately less frequent than framework defaults) --- #
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,  # Keep only the 2 most recent checkpoints - bounds disk usage on a long CPU run.

        # --- Model selection --- #
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,  # Lower WER is better - must be explicit, framework default assumes higher-is-better.

        # --- Misc --- #
        report_to=[],  # No external logging service (wandb/etc.) - avoids needing extra accounts/API keys.
        seed=42,
    )


if __name__ == "__main__":
    args = build_training_arguments()
    print("TrainingArguments built successfully. Key settings:")
    print(f"  use_cpu                     : {args.use_cpu}")
    print(f"  fp16                        : {args.fp16}")
    print(f"  per_device_train_batch_size : {args.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps : {args.gradient_accumulation_steps}")
    print(f"  effective batch size        : {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  gradient_checkpointing      : {args.gradient_checkpointing}")
    print(f"  learning_rate               : {args.learning_rate}")
    print(f"  eval_strategy               : {args.eval_strategy}")
    print(f"  save_total_limit            : {args.save_total_limit}")