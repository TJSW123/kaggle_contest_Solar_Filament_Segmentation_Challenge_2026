"""训练灰度图分割模型，按原图验证，每轮保存 loss / 官方 PQ 曲线。"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from models import build_model
from prepare import ROOT, build_loaders, run_epoch, seed_everything
from self_evaluation import evaluate_model

PROCESSED = ROOT / 'data/train/processed.json'
IMAGE_DIR = ROOT / 'data/train/train_images'
MODEL_DIR = ROOT / 'model'
MODEL_NAME = 'tiny_unet'
EPOCHS = 20
BATCH_SIZE = 4
TILE_SIZE = 512
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2
SEED = 42
DEVICE = 'cuda'
THRESHOLD = 0.5
MIN_AREA = 16


def prepare_training():
    seed_everything(SEED)
    train, val, annotations, loaders = build_loaders(
        PROCESSED, IMAGE_DIR, TILE_SIZE, BATCH_SIZE, VAL_FRACTION,
        SEED, DEVICE,
    )
    print(f'Train: {len(set(r["file_name"] for r in train))} images / {len(loaders[0].dataset)} tiles; '
          f'Validation: {len(set(r["file_name"] for r in val))} images / {len(loaders[2].dataset)} tiles', flush=True)
    return train, val, annotations, loaders


def next_version(model_dir):
    versions = [int(m.group(1)) for p in Path(model_dir).glob('*.pt')
                if (m := re.search(r'_v(\d+)\.pt$', p.name))]
    return max(versions, default=0) + 1


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r['epoch'] for r in history]
    for split, marker in [('train', 'o'), ('val', 'x')]:
        axes[0].plot(epochs, [r[f'{split}_loss'] for r in history], label=split, marker=marker)
        axes[1].plot(epochs, [r[f'{split}_metrics']['pq'] for r in history], label=split, marker=marker)
    axes[0].set_title('Segmentation loss (BCE + Dice)')
    axes[1].set_title('Official-notebook PQ (full images)')
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_xlim(0.5, max(epochs) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_config():
    return dict(
        processed=str(PROCESSED), image_dir=str(IMAGE_DIR), model=MODEL_NAME,
        epochs=EPOCHS, batch_size=BATCH_SIZE, tile_size=TILE_SIZE,
        lr=LEARNING_RATE, val_fraction=VAL_FRACTION, seed=SEED,
        device=DEVICE, threshold=THRESHOLD, min_area=MIN_AREA,
    )


def save_training_results(model, row, history, best, version, stem, config, split):
    quality = (row['val_metrics']['pq'], -row['val_loss'])
    if quality > best:
        best = quality
        checkpoint = dict(
            format_version=1, task='segmentation', model_name=MODEL_NAME,
            version=version, epoch=row['epoch'], state_dict=model.state_dict(),
            config=config, split=split, metrics=row,
        )
        path = MODEL_DIR / f'{stem}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    payload = dict(config=config, split=split, history=history)
    (MODEL_DIR / f'{stem}_history.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8',
    )
    plot_history(history, MODEL_DIR / f'{stem}_history.png')
    return best


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；本项目已禁止回退到 CPU 训练')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    train, val, annotations, (train_loader, train_eval, val_loader) = prepare_training()
    model = build_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version = next_version(MODEL_DIR)
    stem = f'{MODEL_NAME}_v{version:04d}'
    checkpoint_path = MODEL_DIR / f'{stem}.pt'
    config = training_config()
    split = {name: sorted({r['file_name'] for r in records}) for name, records in [('train', train), ('val', val)]}
    history, best = [], (-1.0, float('-inf'))
    for epoch in range(1, EPOCHS + 1):
        prefix = f'Epoch {epoch}/{EPOCHS}'
        run_epoch(model, train_loader, DEVICE, optimizer, f'{prefix} Train')
        row = dict(epoch=epoch,
                   train_loss=run_epoch(
                       model, train_eval, DEVICE, description=f'{prefix} Train loss',
                   ),
                   val_loss=run_epoch(
                       model, val_loader, DEVICE, description=f'{prefix} Validation loss',
                   ))
        for name, records in [('train', train), ('val', val)]:
            row[f'{name}_metrics'] = evaluate_model(
                model, records, annotations, IMAGE_DIR, DEVICE,
                TILE_SIZE, BATCH_SIZE, THRESHOLD, MIN_AREA,
                f'{prefix} {name} PQ',
            )
        history.append(row)
        best = save_training_results(
            model, row, history, best, version, stem, config, split,
        )
        print(f'Epoch {epoch}/{EPOCHS}: train_loss={row["train_loss"]:.4f} '
              f'val_loss={row["val_loss"]:.4f} train_PQ={row["train_metrics"]["pq"]:.4f} '
              f'val_PQ={row["val_metrics"]["pq"]:.4f}', flush=True)
    print(f'已保存本次训练中验证 PQ 最优（同分时 loss 更低）的模型: {checkpoint_path}')


if __name__ == '__main__':
    main()
