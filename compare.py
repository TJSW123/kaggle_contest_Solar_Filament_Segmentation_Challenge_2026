"""用 bbox 填充区域作为监督，比较不同 CNN 的粗定位能力（不是检测框回归）。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models import MODEL_NAMES, build_model
from pipeline import ROOT
from train import common_arguments, prepare, run_epoch
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.set_defaults(epochs=3, max_images=40)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/compare')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.models:
        # 每个模型使用同样的划分、随机种子、batch 顺序、训练轮数及损失函数。
        train, val, _, (training, train_eval, validation) = prepare(args, 'bbox')
        model = build_model(name).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            loss = run_epoch(model, training, args.device, optimizer, args.max_batches)
            print(f'{name} epoch {epoch}/{args.epochs}: bbox training loss={loss:.4f}', flush=True)
        results.append(dict(model=name,
                            train_loss=run_epoch(model, train_eval, args.device, max_batches=args.max_batches),
                            val_loss=run_epoch(model, validation, args.device, max_batches=args.max_batches),
                            parameters=sum(p.numel() for p in model.parameters())))
        del model, optimizer, training, train_eval, validation
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    for offset, key, label in [(-0.18, 'train_loss', 'Train'), (0.18, 'val_loss', 'Validation')]:
        bars = ax.bar(x + offset, [r[key] for r in results], 0.36, label=label)
        ax.bar_label(bars, fmt='%.4f')
    ax.set_xticks(x, [r['model'] for r in results])
    ax.set_ylabel('BCE + Dice loss on filled bounding boxes')
    ax.set_title(f'Coarse localization comparison: {args.epochs} epochs')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / 'comparison.png', dpi=160)
    plt.close(fig)
    payload = dict(results=results, recommended_model=min(results, key=lambda r: r['val_loss'])['model'],
                   config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   split={key: sorted({r['file_name'] for r in rows}) for key, rows in [('train', train), ('val', val)]})
    (args.output_dir / 'comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'按验证 bbox loss 建议初选: {payload["recommended_model"]}; 仍需用分割 PQ 验证。')
    print(args.output_dir / 'comparison.png')


if __name__ == '__main__':
    main()
