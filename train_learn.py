"""固定 ResNet34 U-Net，用 1e-3 到 1e-5 的 5 个学习率各训练一次。"""
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
MODEL_NAME = 'resnet34_unet'
EPOCHS = 20
BATCH_SIZE = 2
TILE_SIZE = 512
LEARNING_RATES = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
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


def lr_tag(learning_rate):
    return f'{learning_rate:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def style_axis(ax, epochs):
    ax.set_xlabel('Epoch')
    ax.set_xlim(0.5, max(epochs) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(alpha=0.3)


def plot_run(history, stem):
    epochs = [row['epoch'] for row in history]
    loss_fig, loss_ax = plt.subplots(figsize=(6, 4))
    pq_fig, pq_ax = plt.subplots(figsize=(6, 4))
    for split, marker in [('train', 'o'), ('val', 'x')]:
        loss_ax.plot(epochs, [row[f'{split}_loss'] for row in history], label=split, marker=marker)
        pq_ax.plot(epochs, [row[f'{split}_metrics']['pq'] for row in history], label=split, marker=marker)
    loss_ax.set_title('Segmentation loss (Focal BCE + Tversky)')
    pq_ax.set_title('Official-notebook PQ (full images)')
    pq_ax.set_ylim(0, 1.02)
    style_axis(loss_ax, epochs)
    style_axis(pq_ax, epochs)
    loss_fig.tight_layout()
    pq_fig.tight_layout()
    loss_fig.savefig(MODEL_DIR / f'{stem}_loss.png', dpi=160)
    pq_fig.savefig(MODEL_DIR / f'{stem}_pq.png', dpi=160)
    plt.close(loss_fig)
    plt.close(pq_fig)


def plot_comparison(results, version):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for result in results:
        epochs = [row['epoch'] for row in result['history']]
        label = result['lr_tag']
        axes[0].plot(epochs, [row['val_loss'] for row in result['history']], label=label, marker='o')
        axes[1].plot(epochs, [row['val_metrics']['pq'] for row in result['history']], label=label, marker='o')
    axes[0].set_title('Validation loss')
    axes[1].set_title('Validation PQ')
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        style_axis(ax, [row['epoch'] for row in results[0]['history']])
    fig.suptitle(f'{MODEL_NAME} learning-rate sweep v{version:04d}')
    fig.tight_layout()
    fig.savefig(MODEL_DIR / f'{MODEL_NAME}_lr_sweep_v{version:04d}.png', dpi=160)
    plt.close(fig)


def training_config(learning_rate):
    return dict(
        processed=str(PROCESSED), image_dir=str(IMAGE_DIR), model=MODEL_NAME,
        encoder_weights='ImageNet1K_DEFAULT',
        epochs=EPOCHS, batch_size=BATCH_SIZE, tile_size=TILE_SIZE,
        lr=learning_rate, learning_rates=list(LEARNING_RATES),
        val_fraction=VAL_FRACTION, seed=SEED,
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
    plot_run(history, stem)
    return best


def train_one(learning_rate, version):
    tag = lr_tag(learning_rate)
    stem = f'{MODEL_NAME}_{tag}_v{version:04d}'
    print(f'开始训练 {MODEL_NAME} lr={learning_rate:g} -> {stem}', flush=True)
    train, val, annotations, loaders = prepare_training()
    split = {name: sorted({r['file_name'] for r in records}) for name, records in [('train', train), ('val', val)]}
    model = build_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader, train_eval, val_loader = loaders
    config = training_config(learning_rate)
    history, best = [], (-1.0, float('-inf'))
    for epoch in range(1, EPOCHS + 1):
        prefix = f'{tag} Epoch {epoch}/{EPOCHS}'
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
        print(f'{tag} Epoch {epoch}/{EPOCHS}: train_loss={row["train_loss"]:.4f} '
              f'val_loss={row["val_loss"]:.4f} train_PQ={row["train_metrics"]["pq"]:.4f} '
              f'val_PQ={row["val_metrics"]["pq"]:.4f}', flush=True)
    del model, optimizer, train_loader, train_eval, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(lr=learning_rate, lr_tag=tag, stem=stem, history=history)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；本项目已禁止回退到 CPU 训练')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'固定模型 {MODEL_NAME}，学习率 {", ".join(f"{lr:g}" for lr in LEARNING_RATES)}', flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version = next_version(MODEL_DIR)
    results = [train_one(learning_rate, version) for learning_rate in LEARNING_RATES]
    plot_comparison(results, version)
    summary = dict(
        model=MODEL_NAME, version=version, learning_rates=list(LEARNING_RATES),
        results=[
            dict(lr=result['lr'], stem=result['stem'], history=result['history'])
            for result in results
        ],
    )
    (MODEL_DIR / f'{MODEL_NAME}_lr_sweep_v{version:04d}.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8',
    )
    print('五次训练完成，每次已保存 loss 与 PQ 两张曲线:')
    for result in results:
        print(f'  lr={result["lr"]:g}: {MODEL_DIR / (result["stem"] + "_loss.png")}')
        print(f'  lr={result["lr"]:g}: {MODEL_DIR / (result["stem"] + "_pq.png")}')


if __name__ == '__main__':
    main()
