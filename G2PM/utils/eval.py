import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, root_mean_squared_error

from ogb.graphproppred import Evaluator


def evaluate(pred, y, params=None):
    metric = params['metric']

    if metric == 'acc':
        return eval_acc(pred, y) * 100
    elif metric == 'auc':
        return eval_auc(pred, y) * 100
    elif metric == 'ap':
        return eval_ap(pred, y) * 100
    elif metric == 'f1':
        return eval_f1(pred, y) * 100
    elif metric == 'rmse':
        return eval_rmse(pred, y)
    elif metric == 'mae':
        return eval_mae(pred, y)
    elif metric == 'hits@20':
        pos_pred, neg_pred = pred, y
        return eval_hits(pos_pred, neg_pred, 20) * 100
    elif metric == 'hits@50':
        pos_pred, neg_pred = pred, y
        return eval_hits(pos_pred, neg_pred, 50) * 100
    elif metric == 'hits@100':
        pos_pred, neg_pred = pred, y
        return eval_hits(pos_pred, neg_pred, 100) * 100
    else:
        raise ValueError(f"Metric {metric} is not supported.")


# Only works for single task classification
def eval_acc(y_pred, y_true):
    device = y_pred.device
    y_true = y_true.to(device)
    num_classes = y_pred.size(1)

    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    return evaluator(y_pred, y_true).item()


def eval_f1(y_pred, y_true):
    y_pred = y_pred.sigmoid().detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    f1 = f1_score(y_true, y_pred > 0.5, average='micro')
    return f1


def eval_auc(y_pred, y_true):
    if len(y_true.shape) == 1:
        y_pred = y_pred.detach().detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        roc = roc_auc_score(y_true, y_pred)
        return roc

    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


def eval_ap(y_pred, y_true):
    if len(y_true.shape) == 1:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        ap = average_precision_score(y_true, y_pred)
        return ap

    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    ap_list = []

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


def eval_rmse(y_pred, y_true):
    if len(y_true.shape) == 1:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        return rmse

    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return sum(rmse_list) / len(rmse_list)


def eval_mae(y_pred, y_true):
    if len(y_true.shape) == 1:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        mae = np.abs(y_true - y_pred).mean()
        return mae

    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    mae_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        mae_list.append(np.abs(y_true[is_labeled, i] - y_pred[is_labeled, i]).mean())

    return sum(mae_list) / len(mae_list)


def eval_hits(pos_pred, neg_pred, K):
    kth_score_in_negative_edges = torch.topk(neg_pred, K)[0][-1]
    hitsK = float(torch.sum(pos_pred > kth_score_in_negative_edges).cpu()) / len(pos_pred)
    return hitsK
