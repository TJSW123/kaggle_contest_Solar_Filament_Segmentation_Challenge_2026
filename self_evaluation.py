"""官方 Self_Evaluation_Notebook v6 的评分适配。

来源：https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
核对日期：2026-09-29。保留 IoU > 0.5、按标注者分别评分、全局累计 PQ
的计算规则；用 COCO RLE 计算重叠，避免展开大量 2048×2048 float32 实例。
没有改用 Hungarian 匹配；多对多计数遵循 notebook 的实现。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util


def overlap_matrices(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        iou = np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    else:
        # pycocotools 的顺序是 detections × ground truths。
        iou = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T
    return iou, 2 * iou / (1 + iou)


def summarize(overlap_df):
    tp = fp = fn = splits = merges = 0
    iou_sum = 0.0
    dice_sum = positive_pairs = 0
    for row in overlap_df.itertuples(index=False):
        iou = np.asarray(row.iou_matrix)
        hits = iou > 0.5
        tp += int(hits.sum())
        fp += int((hits.sum(axis=0) == 0).sum())
        fn += int((hits.sum(axis=1) == 0).sum())
        iou_sum += float(iou[hits].sum())
        overlaps = iou > 0
        splits += int((overlaps.sum(axis=1) > 1).sum())
        merges += int((overlaps.sum(axis=0) > 1).sum())
        dice_sum += float(np.asarray(row.dice_matrix)[overlaps].sum())
        positive_pairs += int(overlaps.sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return dict(pq=iou_sum / denominator if denominator else 0.0,
                tp=tp, fp=fp, fn=fn, splits=splits, merges=merges,
                overlap_dice=dice_sum / positive_pairs if positive_pairs else 0.0)


def get_pq_score(overlap_df):
    return summarize(overlap_df)['pq']


def get_overlap_df(gt_df, pred_df, height=2048, width=2048):
    """与 notebook 相同的 DataFrame 接口；GT ID 必须保留 annotator-image 前缀。"""
    columns = {'filament_id', 'segmentation_rle'}
    for frame in (gt_df, pred_df):
        if not columns.issubset(frame.columns):
            raise ValueError(f'CSV 必须包含 {columns}')
        if frame['filament_id'].duplicated().any():
            raise ValueError('filament_id 必须唯一')
    gt_keys = gt_df.filament_id.str.rsplit('_', n=1).str[0]
    pred_keys = pred_df.filament_id.str.rsplit('_', n=1).str[0]
    rows = []
    for key in gt_keys.unique():
        if '-' not in key:
            raise ValueError('GT filament_id 格式必须是 annotator-image_instance')
        stem = key.split('-', 1)[1]
        def rles(frame, selection):
            return [dict(size=[height, width], counts=s.encode('ascii'))
                    for s in frame.loc[selection, 'segmentation_rle']]
        gt = rles(gt_df, gt_keys == key)
        pred = rles(pred_df, pred_keys == stem)
        iou, dice = overlap_matrices(gt, pred)
        rows.append(dict(annotator_image=key, iou_matrix=iou, dice_matrix=dice,
                         n_gt=len(gt), n_pred=len(pred)))
    return pd.DataFrame(rows, columns=['annotator_image', 'iou_matrix', 'dice_matrix', 'n_gt', 'n_pred'])


def evaluate_model(model, records, annotations, image_dir, device,
                   tile_size=512, batch_size=4, threshold=0.5, min_area=16):
    from pipeline import annotation_rle, predict_image, read_gray, instance_rles
    # 同一原图只推理一次，再分别与各标注者对比。
    by_file = {}
    for record in records:
        by_file.setdefault(record['file_name'], []).append(record)
    rows = []
    for filename, entries in by_file.items():
        probability = predict_image(model, read_gray(Path(image_dir) / filename), device, tile_size, batch_size)
        pred = instance_rles(probability, threshold, min_area)
        for record in entries:
            gt = [annotation_rle(a, record['height'], record['width'])
                  for a in annotations[record['image_id']]]
            iou, dice = overlap_matrices(gt, pred)
            rows.append(dict(annotator_image=record['original_id'], iou_matrix=iou,
                             dice_matrix=dice, n_gt=len(gt), n_pred=len(pred)))
    return summarize(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', required=True, type=Path)
    parser.add_argument('--prediction', required=True, type=Path)
    args = parser.parse_args()
    import json
    frame = get_overlap_df(pd.read_csv(args.ground_truth, keep_default_na=False),
                           pd.read_csv(args.prediction, keep_default_na=False))
    print(json.dumps(summarize(frame), indent=2))


if __name__ == '__main__':
    main()
