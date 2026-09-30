# Solar Filament Segmentation：PyTorch 入门基线

本项目使用灰度图、512×512 切块和小型 U-Net 系列网络。默认路径相对于脚本目录，不需要修改源代码。原始数据不会被覆盖。

## 安装和运行

当前验证环境为 Python 3.13、PyTorch 2.12.1（CPU）。`requirements.txt` 记录本次验证的依赖版本；GPU 环境请按自己的 CUDA 环境安装对应 PyTorch。

```bash
python -m pip install -r requirements.txt
python data/data_processsing.py
python compare.py --epochs 3 --max-images 40
python train.py --model tiny_unet --epochs 20 --batch-size 4
python result.py
```

`compare.py` 会打印按 bbox 验证 loss 初选的模型名，可以把该名称传给 `train.py --model`。可选 `tiny_unet`、`unet`、`resunet`。这三个模型都从头训练，不会下载预训练权重。CPU 上全量训练较慢；有可用 GPU 时自动使用 CUDA，也可以显式传 `--device cpu`。

先检查流程是否能运行：

```bash
python train.py --epochs 1 --max-images 2 --max-batches 1 --batch-size 2 --model-dir model/smoke
python compare.py --epochs 1 --max-images 2 --max-batches 1 --batch-size 2 --output-dir reports/smoke
```

这些命令只做运行检查，不能判断模型好坏。正式训练不要使用 `--max-batches`。冒烟模型放在子目录，默认 `result.py` 不会误选它。

## 各文件的作用

- `data/data_processsing.py`：读取原始 COCO JSON，生成 `data/train/processed.json`。每项只有 `image_id`、`segmentation`、`bbox`，`image_id` 保持原始的「标注者-图像名」。
- `compare.py`：默认抽取 40 张原图，统一划分、随机种子、损失函数、训练轮数和 batch 顺序，训练三个网络；输出 `reports/compare/comparison.png` 和包含数值、配置、划分的 `comparison.json`。
- `train.py`：用分割区域训练。每轮训练后，在无随机增强、eval 模式下计算训练及验证 loss，并在拼接后的完整原图上计算训练及验证 PQ。
- `self_evaluation.py`：提供 `evaluate_model`、`get_overlap_df`、`get_pq_score` 接口，返回 PQ、TP/FP/FN、一对多/多对一数量、正重叠配对的平均 Dice。该 Dice 是诊断值，不是排行榜指标。
- `result.py`：选取 `model/` 中数字版本号最高的 `<模型>_v<版本>.pt`，对测试图像预测，生成 `csv/submission_<模型>_v<版本>.csv` 和推理摘要 JSON。
- `models.py` / `pipeline.py`：共享网络结构、切块数据集、损失函数、拼接、连通域和 RLE 编码。

## 数据格式与标注者

合法 JSON 的多个目标应放在数组 `[...]` 中。精简文件的每项严格只有三个字段：

```json
[
  {"image_id": "040301-20140609195854Bh", "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]], "bbox": [10, 10, 10, 10]}
]
```

`segmentation` 使用 COCO 格式：外层是多个 polygon，每个 polygon 为平铺的 `[x1,y1,x2,y2,...]` 坐标。它表示目标填充区域，不是只训练一像素宽的轮廓线。

实际下载的数据包含 **707 张训练原图、1,154 个标注者/原图记录、8,199 个实例、180 张测试图**。原始 image_id 是 `标注者-图像名` 字符串，例如 `040301-20140609195854Bh`。处理后仍使用这个 id；连字符后的部分加上 `.jpeg` 就是图像文件名，尺寸均为 2048×2048。同一张原图的不同标注者因此仍是不同的监督样本。

多个标注者对同一图像的标注作为不同监督样本保留，不把它们合并成多个独立的真实目标。训练中每个标注者样本分别使用自己的 mask；评价时同一张预测图分别与各标注者比较。按**原图文件名**划分训练和验证集，然后切块，保证同一原图及其所有标注者不会跨集合。标注者较多的原图在损失中权重更大，这是该基线的明确选择。

## bbox 初筛与最终分割

`compare.py` 将 bbox `[x,y,width,height]` 填充为二值矩形区域，用 BCE + soft Dice loss 比较粗定位。这是 bbox 区域监督，**不是**目标检测的坐标回归、框 IoU 或 mAP。所有模型使用相同任务，便于初学者比较；bbox 初筛排名不能保证与最终 PQ 排名一致。

比赛测试集没有公开标签，不能计算测试 loss。所有图中 `Validation` 表示从训练集划出的验证集，正式测试集仅用于 `result.py`。训练与比较使用相同 seed 和样本范围才会得到相同划分；compare 默认 40 张而 train 默认全量，二者默认的 loss 不应直接比较。

图像已经是灰度图，输入为 `[B,1,512,512]`，像素除以 255。2048×2048 切成 4×4 共 16 块；同一原图如果有多个标注者，会有多套 16 块监督样本。预测概率按原坐标拼回 2048×2048，再阈值化并计算全图八连通域。边界两侧接触的目标会自动合并，对角接触也会合并。间隔着背景的目标不会强行连接。

这是“语义分割＋连通域”的实例基线：两个真实目标如果相互接触，也可能被错误合并；预测在接缝处断开也可能产生碎片。没有使用额外膨胀或闭运算来强行修补。后续可以研究重叠切块、边界预测或真正的实例分割网络。

## 官方评分和提交格式

评分适配依据 [官方 Self_Evaluation_Notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) 第 6 版（2026-09-29 取得）。保留 `IoU > 0.5`（不含等号）以及 notebook 的命中矩阵计数方式，全部标注者/图像累计后计算：

```text
PQ = 匹配对 IoU 总和 / (TP + 0.5×FP + 0.5×FN)
```

实现使用 pycocotools 的 RLE 重叠计算节省内存，未改成 Hungarian 匹配，也不是先计算每张图的 PQ 再平均。比赛页面提到额外验证，当前代码以公开 notebook 为准，不保证复现未公开的服务端逻辑。

也可以直接评价自己的 CSV：

```bash
python self_evaluation.py --ground-truth gt.csv --prediction pred.csv
```

GT 的 `filament_id` 应为 `标注者-图像名_实例号`，预测为 `图像名_实例号`，与 notebook 相同。仅 GT CSV 中出现的标注者/图像参与这个 CSV 接口评价。

提交 CSV 两列为 `filament_id,segmentation_rle`。RLE 为 **COCO 压缩 counts 字符串**，不是常见的空格分隔起点长度编码；图像尺寸固定 2048×2048。实例编号从 1 开始，每行一个连通域。无目标的图像不伪造占位实例，另在摘要 JSON 中列出；如果全部图像都没有预测，程序报错且不输出空提交文件。请检查摘要中的无预测图像。

提交格式来源：[比赛说明](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)。

## 模型版本与产物

每次训练在 `model/` 中取全局数字版本号加一，例如 `tiny_unet_v0001.pt`、`resunet_v0002.pt`。同一次训练保留验证 PQ 最高的轮次，同分时选验证 loss 更低的轮次。版本代表一次训练运行，**不代表模型一定更好**。`result.py` 按你的要求选择最新版本；也支持 `--checkpoint model/tiny_unet_v0001.pt` 手动指定。

每个版本还有 `<模型>_v<版本>_history.png`（两张子图：训练/验证 loss 和 PQ）及 `_history.json`（每轮其他评价值、配置和原图划分）。阈值与最小连通域面积默认分别为 0.5、16 像素，保存到 checkpoint；推理沿用相同配置。可通过 `--threshold`、`--min-area` 调整，但应依据验证集选择。

每轮全量训练集和验证集的原图 PQ 计算会额外花时间，这是为了输出完整的每轮评价。该实现未加入断点续训、自动调参、混合精度或大规模缓存。
