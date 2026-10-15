"""微调预训练编码器 + U-Net，按原图验证并保存 loss / 官方 PQ 曲线。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from models import build_model
from prepare import (
    FOCAL_FOREGROUND_WEIGHT,
    FOCAL_GAMMA,
    ROOT,
    TVERSKY_FN_WEIGHT,
    TVERSKY_FP_WEIGHT,
    build_loaders,
    run_epoch,
    seed_everything,
)
from self_evaluation import evaluate_model

PROCESSED = ROOT / 'data/train/processed.json'
IMAGE_DIR = ROOT / 'data/train/train_images'
MODEL_DIR = ROOT / 'model'
MODEL_NAME = 'resnet50_unet'
INPUT_VERSION = 7  # None：从 ImageNet 预训练开始；正整数：加载该版本权重继续训练
OUTPUT_VERSION = 8    # 本次写入的版本号；目标文件已存在时拒绝覆盖
EPOCHS = 20
MICRO_BATCH_SIZE = 2
BATCH_SIZE = 16
ACCUM_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
TILE_SIZE = 512
LEARNING_RATE = [3e-6]
VAL_FRACTION = 0.2
SEED = 42
DEVICE = 'cuda'
THRESHOLD = 0.7
MIN_AREA = 8


def prepare_training():
    seed_everything(SEED)
    train, val, annotations, loaders = build_loaders(
        PROCESSED, IMAGE_DIR, TILE_SIZE, MICRO_BATCH_SIZE, VAL_FRACTION,
        SEED, DEVICE,
    )
    print(f'Train: {len(set(r["file_name"] for r in train))} images / {len(loaders[0].dataset)} tiles; '
          f'Validation: {len(set(r["file_name"] for r in val))} images / {len(loaders[2].dataset)} tiles', flush=True)
    return train, val, annotations, loaders


def version_stem(version):
    return f'{MODEL_NAME}_v{version:04d}'


def version_dir():
    return MODEL_DIR / MODEL_NAME.removesuffix('_unet')


def artifact_path(version, suffix):
    return version_dir() / f'{version_stem(version)}{suffix}'


def require_version(name, value, allow_none=False):
    if allow_none and value is None:
        return
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} 必须是正整数')


def prepare_output():
    require_version('INPUT_VERSION', INPUT_VERSION, allow_none=True)
    require_version('OUTPUT_VERSION', OUTPUT_VERSION)
    if INPUT_VERSION == OUTPUT_VERSION:
        raise ValueError('OUTPUT_VERSION 不能与 INPUT_VERSION 相同，否则会覆盖已有版本')
    weights = artifact_path(OUTPUT_VERSION, '.pt')
    curve = artifact_path(OUTPUT_VERSION, '_history.png')
    if weights.exists() or curve.exists():
        raise FileExistsError(f'输出版本已存在，拒绝覆盖: {weights.parent}')
    version_dir().mkdir(parents=True, exist_ok=True)
    return weights


def load_model():
    model = build_model(MODEL_NAME, pretrained=INPUT_VERSION is None)
    if INPUT_VERSION is None:
        print(f'从 ImageNet 预训练权重开始，保存为 {version_stem(OUTPUT_VERSION)}', flush=True)
        return model.to(DEVICE)
    path = artifact_path(INPUT_VERSION, '.pt')
    if not path.is_file():
        raise FileNotFoundError(f'找不到输入模型: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('model_name') != MODEL_NAME:
        raise ValueError(
            f'输入模型是 {checkpoint.get("model_name")}，与 MODEL_NAME={MODEL_NAME} 不一致'
        )
    model.load_state_dict(checkpoint['state_dict'])
    print(f'从 {path} 继续训练，保存为 {version_stem(OUTPUT_VERSION)}', flush=True)
    return model.to(DEVICE)


def plot_history(history, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    epochs = [r['epoch'] for r in history]
    for split, marker in [('train', 'o'), ('val', 'x')]:
        axes[0].plot(epochs, [r[f'{split}_loss'] for r in history], label=split, marker=marker)
        axes[1].plot(epochs, [r[f'{split}_focal_bce_loss'] for r in history], label=split, marker=marker)
        axes[2].plot(epochs, [r[f'{split}_tversky_loss'] for r in history], label=split, marker=marker)
        axes[3].plot(epochs, [r[f'{split}_metrics']['pq'] for r in history], label=split, marker=marker)
    boundaries = [
        history[index]['epoch'] - 0.5
        for index in range(1, len(history))
        if history[index]['lr'] != history[index - 1]['lr']
    ]
    axes[0].set_title('Segmentation loss (Focal BCE + Tversky)')
    axes[1].set_title('Class-balanced Focal BCE loss')
    axes[2].set_title('FP-weighted Tversky loss')
    axes[3].set_title('Official-notebook PQ (full images)')
    axes[3].set_ylim(0, 1.02)
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color='0.5', ls='--', lw=0.8)
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
        encoder_weights='ImageNet1K_DEFAULT',
        epochs=EPOCHS, batch_size=BATCH_SIZE, micro_batch_size=MICRO_BATCH_SIZE,
        accum_steps=ACCUM_STEPS, tile_size=TILE_SIZE,
        lr=LEARNING_RATE, val_fraction=VAL_FRACTION, seed=SEED,
        device=DEVICE, threshold=THRESHOLD, min_area=MIN_AREA,
        loss='focal_bce+tversky',
        focal_gamma=FOCAL_GAMMA,
        focal_foreground_weight=FOCAL_FOREGROUND_WEIGHT,
        tversky_fp_weight=TVERSKY_FP_WEIGHT,
        tversky_fn_weight=TVERSKY_FN_WEIGHT,
        input_version=INPUT_VERSION, output_version=OUTPUT_VERSION,
    )


def save_training_results(model, row, history, best, config, split):
    quality = (row['val_metrics']['pq'], -row['val_loss'])
    if quality > best:
        best = quality
        checkpoint = dict(
            format_version=1, task='segmentation', model_name=MODEL_NAME,
            version=OUTPUT_VERSION, epoch=row['epoch'], state_dict=model.state_dict(),
            config=config, split=split, metrics=row,
        )
        path = artifact_path(OUTPUT_VERSION, '.pt')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    plot_history(history, artifact_path(OUTPUT_VERSION, '_history.png'))
    return best


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；本项目已禁止回退到 CPU 训练')
    if BATCH_SIZE % MICRO_BATCH_SIZE:
        raise ValueError('BATCH_SIZE 必须能被 MICRO_BATCH_SIZE 整除')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    checkpoint_path = prepare_output()
    train, val, annotations, (train_loader, train_eval, val_loader) = prepare_training()
    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE[0])
    config = training_config()
    split = {name: sorted({r['file_name'] for r in records}) for name, records in [('train', train), ('val', val)]}
    history, best = [], (-1.0, float('-inf'))
    global_epoch = 0
    total_epochs = EPOCHS * len(LEARNING_RATE)
    for learning_rate in LEARNING_RATE:
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        print(f'学习率 {learning_rate:g}，在当前权重上继续训练 {EPOCHS} 轮', flush=True)
        for epoch in range(1, EPOCHS + 1):
            global_epoch += 1
            prefix = f'lr={learning_rate:g} {epoch}/{EPOCHS}'
            run_epoch(model, train_loader, DEVICE, optimizer, f'{prefix} Train', ACCUM_STEPS)
            row = dict(epoch=global_epoch, lr=learning_rate)
            for name, loader in [('train', train_eval), ('val', val_loader)]:
                losses = run_epoch(
                    model, loader, DEVICE, description=f'{prefix} {name.title()} loss',
                    return_components=True,
                )
                row[f'{name}_loss'] = losses['loss']
                row[f'{name}_focal_bce_loss'] = losses['focal_bce']
                row[f'{name}_tversky_loss'] = losses['tversky']
            for name, records in [('train', train), ('val', val)]:
                row[f'{name}_metrics'] = evaluate_model(
                    model, records, annotations, IMAGE_DIR, DEVICE,
                    TILE_SIZE, MICRO_BATCH_SIZE, THRESHOLD, MIN_AREA,
                    f'{prefix} {name} PQ',
                )
            history.append(row)
            best = save_training_results(model, row, history, best, config, split)
            print(f'lr={learning_rate:g} Epoch {epoch}/{EPOCHS} '
                  f'({global_epoch}/{total_epochs}): '
                  f'train_loss={row["train_loss"]:.4f} '
                  f'(Focal BCE={row["train_focal_bce_loss"]:.4f}, '
                  f'Tversky={row["train_tversky_loss"]:.4f}) '
                  f'val_loss={row["val_loss"]:.4f} '
                  f'(Focal BCE={row["val_focal_bce_loss"]:.4f}, '
                  f'Tversky={row["val_tversky_loss"]:.4f}) '
                  f'train_PQ={row["train_metrics"]["pq"]:.4f} '
                  f'val_PQ={row["val_metrics"]["pq"]:.4f}', flush=True)
    print(f'已保存本次训练中验证 PQ 最优（同分时 loss 更低）的模型: {checkpoint_path}')


if __name__ == '__main__':
    main()
