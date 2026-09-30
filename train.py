"""训练灰度图分割模型，按原图验证，每轮保存 loss / 官方 PQ 曲线。"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch.utils.data import DataLoader

from models import MODEL_NAMES, build_model
from pipeline import ROOT, TileDataset, load_data, seed_everything, segmentation_loss, split_images
from self_evaluation import evaluate_model


def common_arguments(parser):
    parser.add_argument('--processed', type=Path, default=ROOT / 'data/train/processed.json')
    parser.add_argument('--image-dir', type=Path, default=ROOT / 'data/train/train_images')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-images', type=int, default=0, help='0=所有原图；非零用于小规模练习')
    parser.add_argument('--max-batches', type=int, default=0, help='0=所有 batch；非零仅用于运行检查')


def prepare(args, target):
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.lr <= 0:
        raise ValueError('epochs、batch_size、threads、lr 必须为正数')
    if min(args.max_images, args.max_batches, args.workers) < 0:
        raise ValueError('max_images、max_batches、workers 不能为负数')
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    images, annotations = load_data(args.processed)
    train, val = split_images(images, args.val_fraction, args.seed, args.max_images)
    for record in train + val:
        if not (args.image_dir / record['file_name']).is_file():
            raise FileNotFoundError(args.image_dir / record['file_name'])
    kwargs = dict(annotations=annotations, image_dir=args.image_dir, tile_size=args.tile_size, target=target)
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=args.device.startswith('cuda'))
    loaders = (
        DataLoader(TileDataset(train, augment=True, **kwargs), shuffle=True,
                   generator=torch.Generator().manual_seed(args.seed), **options),
        DataLoader(TileDataset(train, **kwargs), shuffle=False, **options),
        DataLoader(TileDataset(val, **kwargs), shuffle=False, **options))
    print(f'Train: {len(set(r["file_name"] for r in train))} images / {len(loaders[0].dataset)} tiles; '
          f'Validation: {len(set(r["file_name"] for r in val))} images / {len(loaders[2].dataset)} tiles', flush=True)
    if args.max_batches:
        print('注意：max-batches 限制了训练和 loss 计算；数值只用于 smoke test。', flush=True)
    return train, val, annotations, loaders


def run_epoch(model, loader, device, optimizer=None, max_batches=0):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for index, (inputs, target, valid) in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            inputs, target, valid = (x.to(device, non_blocking=True) for x in (inputs, target, valid))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(inputs), target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('loss 出现 NaN/Inf')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
    if not count:
        raise ValueError('数据集为空')
    return total / count


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--model', choices=MODEL_NAMES, default='tiny_unet')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'model')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-area', type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.min_area < 1:
        parser.error('threshold 必须在 (0,1)，min-area 必须为正整数')
    train, val, annotations, (train_loader, train_eval, val_loader) = prepare(args, 'segmentation')
    model = build_model(args.model).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(args.model_dir)
    stem = f'{args.model}_v{version:04d}'
    checkpoint_path = args.model_dir / f'{stem}.pt'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    split = {name: sorted({r['file_name'] for r in records}) for name, records in [('train', train), ('val', val)]}
    history, best = [], (-1.0, float('-inf'))
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, train_loader, args.device, optimizer, args.max_batches)
        row = dict(epoch=epoch,
                   train_loss=run_epoch(model, train_eval, args.device, max_batches=args.max_batches),
                   val_loss=run_epoch(model, val_loader, args.device, max_batches=args.max_batches))
        for name, records in [('train', train), ('val', val)]:
            row[f'{name}_metrics'] = evaluate_model(model, records, annotations, args.image_dir,
                args.device, args.tile_size, args.batch_size, args.threshold, args.min_area)
        history.append(row)
        quality = (row['val_metrics']['pq'], -row['val_loss'])
        if quality > best:
            best = quality
            checkpoint = dict(format_version=1, task='segmentation', model_name=args.model,
                              version=version, epoch=epoch, state_dict=model.state_dict(),
                              config=config, split=split, metrics=row)
            temporary = checkpoint_path.with_suffix('.pt.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        (args.model_dir / f'{stem}_history.json').write_text(json.dumps(
            dict(config=config, split=split, history=history), indent=2), encoding='utf-8')
        plot_history(history, args.model_dir / f'{stem}_history.png')
        print(f'Epoch {epoch}/{args.epochs}: train_loss={row["train_loss"]:.4f} '
              f'val_loss={row["val_loss"]:.4f} train_PQ={row["train_metrics"]["pq"]:.4f} '
              f'val_PQ={row["val_metrics"]["pq"]:.4f}', flush=True)
    print(f'已保存本次训练中验证 PQ 最优（同分时 loss 更低）的模型: {checkpoint_path}')


if __name__ == '__main__':
    main()
