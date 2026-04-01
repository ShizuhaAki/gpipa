import logging
from pathlib import Path

import torch
import torch.nn as nn
from rich.logging import RichHandler
from torch.utils.data import DataLoader, random_split
import yaml
from tqdm import tqdm

from gpipa.acoustic.dataloader import (
    DataBakerDataset,
    FrameWiseDataCollator,
    save_vocab,
)
from gpipa.acoustic.model import FrameWiseBiLSTM

logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

import math
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    return LambdaLR(optimizer, lr_lambda)

def _build_vocab_from_torch_dataset(ds: DataBakerDataset, blank_tokens=None) -> dict[str, int]:
    if blank_tokens is None:
        blank_tokens = {"sil", "sp1", "sp2"}
    else:
        blank_tokens = set(blank_tokens)

    labels = set()
    for rec in ds.records:
        intervals = ds._parse_textgrid(rec["interval_path"])
        for interval in intervals:
            lbl = str(interval["label"])
            if lbl and lbl not in blank_tokens:
                labels.add(lbl)

    tokens = ["sil"] + sorted(labels)
    return {token: idx for idx, token in enumerate(tokens)}


def load_config(config_path: str | Path = "config.yml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader) or {}

    if not config["root_path"]:
        raise ValueError("config.yml must include a non-empty 'root_path'")
    return config


def _run_epoch(model, dataloader, criterion, device, optimizer=None, scheduler=None, max_grad_norm=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_frames = 0
    total_correct = 0

    for batch in tqdm(dataloader):
        fbanks = batch.fbanks.to(device)
        labels = batch.labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(fbanks)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

        total_loss += float(loss.item())

        valid = labels != -100
        if valid.any():
            pred = logits.argmax(dim=-1)
            total_correct += int((pred[valid] == labels[valid]).sum().item())
            total_frames += int(valid.sum().item())

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_frames)
    return avg_loss, accuracy


def train(config_path: str | Path = "config.yml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    logger.info("Stage 1: Data")
    ds = DataBakerDataset(config["root_path"])
    label2id = _build_vocab_from_torch_dataset(ds)
    num_classes = len(label2id)

    train_size = int(len(ds) * config["train_split"])
    eval_size = len(ds) - train_size
    if train_size <= 0 or eval_size <= 0:
        raise ValueError(
            "Invalid train/eval split. Adjust 'train_split' to keep both splits non-empty."
        )

    split_generator = torch.Generator().manual_seed(int(config["seed"]))
    train_ds, eval_ds = random_split(ds, [train_size, eval_size], generator=split_generator)

    collator = FrameWiseDataCollator(
        label2id=label2id,
        sampling_rate=config["sampling_rate"],
        hop_length_ms=config["hop_length_ms"],
        cache_dir=config.get("feature_cache_dir"),
    )

    def collate_fn(features):
        # Keep collator backward-compatible while dataset now returns raw ndarray audio.
        normalized = []
        for item in features:
            if isinstance(item.get("audio"), dict) and "array" in item["audio"]:
                normalized.append(item)
                continue
            new_item = dict(item)
            new_item["audio"] = {"array": item["audio"]}
            normalized.append(new_item)
        return collator(normalized)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["num_worker"],
        pin_memory=pin_memory,
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_worker"],
        pin_memory=pin_memory,
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if config["save_vocab"]:
        save_vocab(label2id, output_dir / "vocab.json")

    logger.info("Stage 2: Model")
    model = FrameWiseBiLSTM(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=len(label2id),
        dropout=config["dropout"],
    ).to(device)
    logger.info("Model initialized with %d parameters", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_dataloader) * config["epochs"]
    warmup_steps = int(total_steps * 0.1) 
    
    scheduler = get_warmup_cosine_schedule(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    logger.info("Note: Using weighted class")
    class_weights = torch.ones(num_classes)
    class_weights[0] = 0.05
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)

    logger.info(
        "Stage 3: Train (%d examples train / %d examples eval)",
        len(train_ds),
        len(eval_ds),
    )
    best_eval_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_dataloader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=config["max_grad_norm"],
        )
        eval_loss, eval_acc = _run_epoch(
            model,
            eval_dataloader,
            criterion,
            device,
            optimizer=None,
            max_grad_norm=config["max_grad_norm"],
        )

        logger.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f | eval_loss=%.4f eval_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            eval_loss,
            eval_acc,
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "label2id": label2id,
                    "config": config,
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                },
                ckpt_path,
            )
            logger.info("Saved new best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    train()
        