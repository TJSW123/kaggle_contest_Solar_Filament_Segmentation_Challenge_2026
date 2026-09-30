"""用真实分割区域训练，按完整原图的验证 PQ 比较不同 CNN。"""
import json

import numpy as np
import torch

from models import MODEL_NAMES, build_model
from prepare import ROOT, build_loaders, run_epoch, seed_everything
from self_evaluation import evaluate_model
from matplotlib import pyplot as plt

PROCESSED = ROOT / 'data/train/processed.json'
IMAGE_DIR = ROOT / 'data/train/train_images'
OUTPUT_DIR = ROOT / 'reports/compare'
MODELS = MODEL_NAMES
EPOCHS = 5
BATCH_SIZE = 4
TILE_SIZE = 512
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2
SEED = 42
DEVICE = 'cuda'
THRESHOLD = 0.5
MIN_AREA = 16


def comparison_config():
    return dict(
        processed=str(PROCESSED), image_dir=str(IMAGE_DIR),
        models=list(MODELS), output_dir=str(OUTPUT_DIR), epochs=EPOCHS,
        batch_size=BATCH_SIZE, tile_size=TILE_SIZE, lr=LEARNING_RATE,
        val_fraction=VAL_FRACTION, seed=SEED, device=DEVICE,
        threshold=THRESHOLD, min_area=MIN_AREA,
    )


def plot_comparison(results):
    fig, (pq_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(results))
    labels = [f'{row["model"]}\n(best epoch {row["best_epoch"]})' for row in results]

    bars = pq_ax.bar(x, [row['val_metrics']['pq'] for row in results])
    pq_ax.bar_label(bars, fmt='%.4f')
    pq_ax.set_xticks(x, labels)
    pq_ax.set_ylabel('Validation PQ on full images')
    pq_ax.set_ylim(0, 1.05)
    pq_ax.set_title('Higher PQ is better')

    for offset, key, label in [(-0.18, 'train_loss', 'Train'), (0.18, 'val_loss', 'Validation')]:
        bars = loss_ax.bar(x + offset, [row[key] for row in results], 0.36, label=label)
        loss_ax.bar_label(bars, fmt='%.4f')
    loss_ax.set_xticks(x, labels)
    loss_ax.set_ylabel('Segmentation BCE + Dice loss')
    loss_ax.set_title('Loss at the selected epoch')
    loss_ax.legend()

    fig.suptitle(f'Segmentation comparison: {EPOCHS} epochs per model')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'comparison.png', dpi=160)
    plt.close(fig)


def save_comparison(results, train, val):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recommended = max(
        results, key=lambda row: (row['val_metrics']['pq'], -row['val_loss'])
    )['model']
    payload = dict(
        task='segmentation',
        selection_metric='val_pq_then_lower_val_loss',
        results=results,
        recommended_model=recommended,
        config=comparison_config(),
        split={
            name: sorted({record['file_name'] for record in records})
            for name, records in [('train', train), ('val', val)]
        },
    )
    (OUTPUT_DIR / 'comparison.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8',
    )
    plot_comparison(results)
    return recommended


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；本项目已禁止回退到 CPU 训练')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    results = []
    for name in MODELS:
        # 每个模型使用同样的划分、随机种子、batch 顺序、训练轮数及损失函数。
        seed_everything(SEED)
        train, val, annotations, loaders = build_loaders(
            PROCESSED, IMAGE_DIR, TILE_SIZE, BATCH_SIZE, VAL_FRACTION,
            SEED, DEVICE,
        )
        training, train_eval, validation = loaders
        print(f'{name}: {len({r["file_name"] for r in train})} train images / '
              f'{len({r["file_name"] for r in val})} validation images', flush=True)
        model = build_model(name).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        history, best_row = [], None
        for epoch in range(1, EPOCHS + 1):
            prefix = f'{name} {epoch}/{EPOCHS}'
            run_epoch(model, training, DEVICE, optimizer, f'{prefix} Train')
            row = dict(epoch=epoch,
                       train_loss=run_epoch(
                           model, train_eval, DEVICE, description=f'{prefix} Train loss',
                       ),
                       val_loss=run_epoch(
                           model, validation, DEVICE, description=f'{prefix} Validation loss',
                       ),
                       val_metrics=evaluate_model(
                           model, val, annotations, IMAGE_DIR, DEVICE,
                           TILE_SIZE, BATCH_SIZE, THRESHOLD, MIN_AREA,
                           f'{prefix} Validation PQ',
                       ))
            history.append(row)
            if best_row is None or (row['val_metrics']['pq'], -row['val_loss']) > (
                    best_row['val_metrics']['pq'], -best_row['val_loss']):
                best_row = row
            print(f'{name} epoch {epoch}/{EPOCHS}: train_loss={row["train_loss"]:.4f} '
                  f'val_loss={row["val_loss"]:.4f} val_PQ={row["val_metrics"]["pq"]:.4f}', flush=True)
        results.append(dict(model=name, best_epoch=best_row['epoch'],
                            train_loss=best_row['train_loss'], val_loss=best_row['val_loss'],
                            val_metrics=best_row['val_metrics'], history=history,
                            parameters=sum(p.numel() for p in model.parameters())))
        del model, optimizer, training, train_eval, validation
    recommended = save_comparison(results, train, val)
    print(f'按验证分割 PQ 初选（同分时 loss 更低）: {recommended}; '
          '仍需更充分训练确认排名。')
    print(OUTPUT_DIR / 'comparison.png')


if __name__ == '__main__':
    main()
