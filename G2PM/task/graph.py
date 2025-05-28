import yaml
import wandb
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.linkproppred import *
from ogb.nodeproppred import *

from model.random_walk import get_patterns_for_graph

from utils.eval import evaluate
from utils.utils import get_device_from_model, seed_everything, check_path, get_num_params, to_millions, mask2idx


def multitask_cross_entropy(y_pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    y[y == 0] = -1
    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def multitask_regression(y_pred, y, metric='rmse'):
    if metric == 'rmse':
        criterion = nn.MSELoss(reduction="none")
    elif metric == 'mae':
        criterion = nn.L1Loss(reduction="none")

    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred, exist_y)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def preprocess_graph(datasets, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']
    pattern_size = params['pattern_size']
    p = params['p']
    q = params['q']

    pattern_dir = osp.join(params['pattern_path'], params['dataset'])
    pattern_dict = {}

    if isinstance(datasets, dict):
        for key, subset in datasets.items():
            cur_dir = osp.join(pattern_dir, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}", key)
            check_path(cur_dir)

            pattern_path = osp.join(cur_dir, f"ptn.pt")
            nid_path = osp.join(cur_dir, f"nid.pt")
            eid_path = osp.join(cur_dir, f"eid.pt")

            if osp.exists(pattern_path) and osp.exists(nid_path) and osp.exists(eid_path):
                patterns = torch.load(pattern_path)
                nids = torch.load(nid_path)
                eids = torch.load(eid_path)
            else:
                patterns, nids, eids = get_patterns_for_graph(subset, params)
                torch.save(patterns, pattern_path)
                torch.save(nids, nid_path)
                torch.save(eids, eid_path)

            pattern_dict[key] = {'pattern': patterns, 'nid': nids, 'eid': eids}
    else:
        cur_dir = osp.join(pattern_dir, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
        check_path(cur_dir)

        pattern_path = osp.join(cur_dir, f"ptn.pt")
        nid_path = osp.join(cur_dir, f"nid.pt")
        eid_path = osp.join(cur_dir, f"eid.pt")

        if osp.exists(pattern_path) and osp.exists(nid_path) and osp.exists(eid_path):
            patterns = torch.load(pattern_path, weights_only=True)
            nids = torch.load(nid_path, weights_only=True)
            eids = torch.load(eid_path, weights_only=True)

        else:
            patterns, nids, eids = get_patterns_for_graph(datasets, params)
            torch.save(patterns, pattern_path)
            torch.save(nids, nid_path)
            torch.save(eids, eid_path)

        pattern_dict = {'pattern': patterns, 'nid': nids, 'eid': eids}

    return pattern_dict


def train_graph(dataset, model, optimizer, split=None, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    bs = params['batch_size']

    total_loss, total_val_loss, total_test_loss = 0, 0, 0

    if isinstance(split, int):
        dataset = dataset['train'] if params['split'] != 'pretrain' else dataset['full']
        num_graphs = len(dataset)
        graphs = torch.arange(num_graphs)
    else:
        dataset = dataset
        graphs = mask2idx(split['train'])
        num_graphs = len(graphs)

    y = dataset.y

    # We do batch training by default
    num_batches = (num_graphs + bs - 1) // bs
    train_perm = torch.randperm(num_graphs)

    for i in range(num_batches):
        cur_graphs = graphs[train_perm[i * bs: (i + 1) * bs]]
        cur_y = y[cur_graphs].to(device)

        pred, instance_emb, pattern_emb, commit_loss = model(dataset, cur_graphs, params, mode='train')
        if y.ndim == 1:
            if params['metric'] == 'rmse':
                loss = F.mse_loss(pred.squeeze(), cur_y.float())
            elif params['metric'] == 'mae':
                loss = F.l1_loss(pred.squeeze(), cur_y.float())
            else:
                loss = F.cross_entropy(pred, cur_y, label_smoothing=params['label_smoothing'])
        else:
            if params['metric'] in ['rmse', 'mae']:
                loss = multitask_regression(pred, cur_y.float(), metric=params['metric'])
            else:
                loss = multitask_cross_entropy(pred, cur_y)

        loss = loss + commit_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    total_loss /= num_batches
    total_val_loss /= num_batches
    total_test_loss /= num_batches

    return {'train': total_loss, 'val': total_val_loss, 'test': total_test_loss}


def eval_graph(graph, model, split=None, params=None):
    model.eval()
    bs = params['batch_size']

    results = {}
    results['metric'] = params['metric']
    results['train'] = 0

    with torch.no_grad():
        for key in ['val', 'test']:
            if isinstance(split, int):
                dataset = graph[key]
                num_graphs = len(dataset)
                graphs = torch.arange(num_graphs)
            else:
                dataset = graph
                graphs = mask2idx(split[key])
                num_graphs = len(graphs)

            y = dataset.y[graphs]

            num_batches = (num_graphs + bs - 1) // bs

            pred_list = []
            for i in range(num_batches):
                cur_graphs = graphs[i * bs: (i + 1) * bs]
                pred, _, _, _ = model(dataset, cur_graphs, params, mode=key)
                pred_list.append(pred.detach())
            pred = torch.cat(pred_list, dim=0)

            value = evaluate(pred, y, params=params)
            results[key] = value

    return results



def pretrain_graph(dataset, model, optimizer, scheduler=None, params=None):
    if params['inference_only']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    
    bs = params['batch_size']
    graphs = torch.arange(len(dataset))
    if params['train_mask'] is not None:
        graphs = graphs[params['train_mask']]
    num_graphs = graphs.size(0)
    graphs = graphs[torch.randperm(num_graphs)]
    num_batches = (num_graphs + bs - 1) // bs

    total_loss = 0
    for i in range(num_batches):
        cur_graphs = graphs[i * bs: (i + 1) * bs]
        loss = model.pretrain_graph(dataset, cur_graphs, params)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if params['objective_on'] == 'emb':
            if i % params['ema_update_every'] == 0:
                model.ema_update(alpha=params['ema_alpha'])

        wandb.log({
            "training dynamics/step-wise train_loss": loss.item(),
            "training dynamics/step-wise lr": scheduler.get_last_lr()[0],
        })
    total_loss /= num_batches

    return {'train': total_loss, 'val': 0, 'test': 0}


def linear_probe_graph(graph, model, splits, params):
    model.eval()
    device = get_device_from_model(model)

    bs = params['batch_size']
    embeddings_list = []

    if isinstance(graph, dict):
        full_dataset = graph['full']
    else:
        full_dataset = graph

    with torch.no_grad():
        graphs = torch.arange(len(full_dataset))
        num_graphs = graphs.size(0)
        num_batches = (num_graphs + bs - 1) // bs
        for i in range(num_batches):
            cur_graphs = graphs[i * bs: (i + 1) * bs]
            _, instance_emb, _, _ = model(full_dataset, cur_graphs, params, mode='eval')
            embeddings_list.append(instance_emb.detach().cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    y = full_dataset.y[graphs]

    # Check if embeddings are collapsed
    emb_mean = embeddings.mean(dim=0)
    emb_std = embeddings.std(dim=0)
    wandb.log({
        "If embeddings are collapsed/emb_mean_scalar": emb_mean.mean().item(),
        "If embeddings are collapsed/emb_std_scalar": emb_std.mean().item(),
        "If embeddings are collapsed/emb_mean_min": emb_mean.min().item(),
        "If embeddings are collapsed/emb_mean_max": emb_mean.max().item(),
        "If embeddings are collapsed/emb_std_min": emb_std.min().item(), 
        "If embeddings are collapsed/emb_std_max": emb_std.max().item()
    })

    best_val_accs = []
    best_test_accs = []

    for split in splits:
        # Train linear classifier
        train_mask = split['train']
        val_mask = split['val']
        test_mask = split['test']

        X_train = embeddings[train_mask]
        y_train = y[train_mask]

        classifier = nn.Linear(embeddings.shape[1], params['output_dim']).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=params['linear_probe_lr'], weight_decay=params['linear_probe_weight_decay'])

        # Train for a fixed number of epochs
        num_epochs = params['linear_probe_epochs']
        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(num_epochs):
            classifier.train()
            optimizer.zero_grad()

            # Training in mini-batches
            train_batch_size = 200000
            train_idx = torch.randperm(X_train.size(0))
            train_loss = 0
            num_train_batches = (X_train.size(0) + train_batch_size - 1) // train_batch_size

            for i in range(num_train_batches):
                batch_idx = train_idx[i * train_batch_size:(i + 1) * train_batch_size]
                batch_X = X_train[batch_idx].to(device)
                batch_y = y_train[batch_idx].to(device)

                batch_logits = classifier(batch_X)
                
                if y.ndim == 1:
                    if params['metric'] == 'rmse':
                        loss = F.mse_loss(batch_logits.squeeze(), batch_y.float())
                    elif params['metric'] == 'mae':
                        loss = F.l1_loss(batch_logits.squeeze(), batch_y.float())
                    else:
                        loss = F.cross_entropy(batch_logits, batch_y)
                else:
                    if params['metric'] in ['rmse', 'mae']:
                        loss = multitask_regression(batch_logits, batch_y.float(), metric=params['metric'])
                    else:
                        loss = multitask_cross_entropy(batch_logits, batch_y)

                loss.backward()
                train_loss += loss.item()

            optimizer.step()

            # Evaluation in mini-batches
            classifier.eval()
            val_batch_size = 200000
            test_batch_size = 200000

            with torch.no_grad():
                # Validation
                val_logits_list = []
                num_val_batches = (embeddings[val_mask].size(0) + val_batch_size - 1) // val_batch_size
                for i in range(num_val_batches):
                    start_idx = i * val_batch_size
                    end_idx = min((i + 1) * val_batch_size, embeddings[val_mask].size(0))
                    batch_logits = classifier(embeddings[val_mask][start_idx:end_idx].to(device))
                    val_logits_list.append(batch_logits)
                val_logits = torch.cat(val_logits_list, dim=0)

                # Test
                test_logits_list = []
                num_test_batches = (embeddings[test_mask].size(0) + test_batch_size - 1) // test_batch_size
                for i in range(num_test_batches):
                    start_idx = i * test_batch_size
                    end_idx = min((i + 1) * test_batch_size, embeddings[test_mask].size(0))
                    batch_logits = classifier(embeddings[test_mask][start_idx:end_idx].to(device))
                    test_logits_list.append(batch_logits)
                test_logits = torch.cat(test_logits_list, dim=0)

                val_acc = evaluate(val_logits, y[val_mask].to(device), params=params)
                test_acc = evaluate(test_logits, y[test_mask].to(device), params=params)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
            print('val acc: {:.4f}, test acc: {:.4f}'.format(val_acc, test_acc))
        best_val_accs.append(best_val_acc)
        best_test_accs.append(best_test_acc)

    return {'train': 0, 'train_std': 0, 'val': np.mean(best_val_accs), 'val_std': np.std(best_val_accs),
            'test': np.mean(best_test_accs), 'test_std': np.std(best_test_accs), 'metric': params['metric']}


def svm_probe_graph(graph, model, splits, params):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.svm import SVC
    
    model.eval()
    device = get_device_from_model(model)

    bs = params['batch_size']
    embeddings_list = []

    if isinstance(graph, dict):
        full_dataset = graph['full']
    else:
        full_dataset = graph

    with torch.no_grad():
        graphs = torch.arange(len(full_dataset))
        num_graphs = graphs.size(0)
        num_batches = (num_graphs + bs - 1) // bs
        for i in range(num_batches):
            cur_graphs = graphs[i * bs: (i + 1) * bs]
            _, instance_emb, _, _ = model(full_dataset, cur_graphs, params, mode='eval')
            embeddings_list.append(instance_emb.detach().cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    y = full_dataset.y[graphs]

    # Check if embeddings are collapsed
    emb_mean = embeddings.mean(dim=0)
    emb_std = embeddings.std(dim=0)
    wandb.log({
        "If embeddings are collapsed/emb_mean_scalar": emb_mean.mean().item(),
        "If embeddings are collapsed/emb_std_scalar": emb_std.mean().item(),
        "If embeddings are collapsed/emb_mean_min": emb_mean.min().item(),
        "If embeddings are collapsed/emb_mean_max": emb_mean.max().item(),
        "If embeddings are collapsed/emb_std_min": emb_std.min().item(), 
        "If embeddings are collapsed/emb_std_max": emb_std.max().item()
    })

    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    embeddings = embeddings.cpu().numpy()
    y = y.cpu().numpy()

    acc_list = []
    for train_index, test_index in kf.split(embeddings, y):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = y[train_index], y[test_index]

        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}

        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params, n_jobs=16)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        acc_list.append(acc)

    return {'train': 0, 'train_std': 0, 'val': np.mean(acc_list), 'val_std': np.std(acc_list),
            'test': np.mean(acc_list), 'test_std': np.std(acc_list), 'metric': 'acc'}