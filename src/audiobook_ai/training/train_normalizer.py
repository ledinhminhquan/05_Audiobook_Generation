"""Fine-tune the ByT5/T5 Text-Normalization model (HF Seq2SeqTrainer).

Resume-safe (``get_last_checkpoint``), bf16/tf32 on H100/A100, char-level ByT5 by
default. ``compute_metrics`` reports sentence-level exact-match accuracy plus a
macro per-semiotic-class accuracy. Heavy imports are lazy so the module imports
on core deps; training itself obviously needs torch/transformers/datasets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..models import model_registry as reg
from ..data.dataset import corpus_signature, load_or_build, to_hf_datasets

logger = get_logger(__name__)


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _make_compute_metrics(tokenizer, eval_rows: List[Dict]):
    import numpy as np

    classes_per_row = [r.get("classes", []) for r in eval_rows]

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        correct = [_ws(p) == _ws(l) for p, l in zip(dec_preds, dec_labels)]
        sent_acc = float(np.mean(correct)) if correct else 0.0
        # macro per-class accuracy (aligned by index with the eval dataset)
        per_class_hits: Dict[str, List[int]] = {}
        for i, ok in enumerate(correct):
            if i >= len(classes_per_row):
                break
            for c in (classes_per_row[i] or ["PLAIN"]):
                per_class_hits.setdefault(c, []).append(1 if ok else 0)
        per_class = {f"acc_{c}": float(np.mean(v)) for c, v in per_class_hits.items()}
        macro = float(np.mean(list(per_class.values()))) if per_class else sent_acc
        return {"sentence_accuracy": sent_acc, "macro_class_accuracy": macro, **per_class}

    return compute_metrics


def train_normalizer(cfg: AppConfig, limit: Optional[int] = None, resume: bool = True,
                     base_model: Optional[str] = None) -> Dict:
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    mc = cfg.model
    model_id = base_model or mc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)
    torch.backends.cudnn.allow_tf32 = bool(mc.tf32)

    splits = load_or_build(cfg)
    if limit:
        for k in ("train", "val"):
            splits[k] = splits[k][:limit]
    ds = to_hf_datasets(splits)
    logger.info("Training %s on %d examples (val %d)", model_id, len(ds["train"]), len(ds.get("val", [])))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if mc.dropout_rate is not None and hasattr(model.config, "dropout_rate"):
        model.config.dropout_rate = mc.dropout_rate

    prefix = mc.task_prefix

    def preprocess(batch):
        inputs = [prefix + t for t in batch["before"]]
        model_inputs = tokenizer(inputs, max_length=mc.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["after"], max_length=mc.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    keep = ds["train"].column_names
    tokenized = ds.map(preprocess, batched=True, remove_columns=keep, desc="tokenize")
    eval_rows = splits.get("val", [])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    out_dir = mc.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=False,
        num_train_epochs=mc.num_train_epochs,
        learning_rate=mc.learning_rate,
        per_device_train_batch_size=mc.per_device_train_batch_size,
        per_device_eval_batch_size=mc.per_device_eval_batch_size,
        gradient_accumulation_steps=mc.gradient_accumulation_steps,
        warmup_ratio=mc.warmup_ratio,
        weight_decay=mc.weight_decay,
        label_smoothing_factor=mc.label_smoothing_factor,
        max_grad_norm=mc.max_grad_norm,
        lr_scheduler_type="cosine",
        bf16=bool(mc.bf16),
        fp16=bool(mc.fp16),
        gradient_checkpointing=bool(mc.gradient_checkpointing),
        group_by_length=bool(mc.group_by_length),
        predict_with_generate=True,
        generation_max_length=mc.max_target_length,
        generation_num_beams=mc.generation_num_beams,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=mc.eval_steps,
        save_steps=mc.save_steps,
        logging_steps=mc.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="sentence_accuracy",
        greater_is_better=True,
        seed=mc.seed,
        report_to=[],
        dataloader_num_workers=2,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("val"),
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_make_compute_metrics(tokenizer, eval_rows),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=mc.early_stopping_patience)],
    )

    last_ckpt = get_last_checkpoint(str(out_dir)) if resume and out_dir.exists() else None
    if last_ckpt:
        logger.info("Resuming from checkpoint %s", last_ckpt)
    trainer.train(resume_from_checkpoint=last_ckpt)

    metrics = trainer.evaluate()
    # persist best model under a versioned dir + update the 'latest' pointer
    version = reg.make_version(model_id)
    final_dir = out_dir / version
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature=corpus_signature(cfg.data),
                       metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    reg.update_latest_pointer(out_dir, final_dir)
    (out_dir / "last_metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    logger.info("Training done. sentence_accuracy=%.4f -> %s", metrics.get("eval_sentence_accuracy", 0.0), final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id,
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}


__all__ = ["train_normalizer"]
