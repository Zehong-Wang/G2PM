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

from model.random_walk import get_patterns

from utils.eval import evaluate
from utils.utils import get_device_from_model, seed_everything, check_path, get_num_params, to_millions, mask2idx


def multitask_cross_entropy(y_pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    loss = 0.0
    for idx in range(y.shape[1]):
        cur_y = y[:, idx]
        cur_pred = y_pred[:, idx]
        task_loss = criterion(cur_pred.double(), cur_y)
        loss += torch.mean(task_loss)

    return loss / y.shape[1]


def preprocess_node(graph, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']
    pattern_size = params['pattern_size']
    p = params['p']
    q = params['q']

    if isinstance(graph, dict):
        pattern_dir = osp.join(params['pattern_path'], params['dataset'])
        pattern_dict = {}
        for key, g_set in graph.items():
            cur_dir = osp.join(pattern_dir, key, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
            check_path(cur_dir)
            pattern_dict[key] = {}

            for i, g in enumerate(g_set):
                pattern_path = osp.join(cur_dir, f"ptn_{i}.pt")
                eid_path = osp.join(cur_dir, f"eid_{i}.pt")
                if osp.exists(pattern_path) and osp.exists(eid_path):
                    patterns = torch.load(pattern_path, map_location=torch.device('cpu'))
                    eids = torch.load(eid_path, map_location=torch.device('cpu'))
                else:
                    patterns, eids = get_patterns(g, params)
                    torch.save(patterns, pattern_path)
                    torch.save(eids, eid_path)
                pattern_dict[key][i] = {'pattern': patterns, 'eid': eids}
        return pattern_dict
    else:
        pattern_dir = osp.join(params['pattern_path'], params['dataset'])
        check_path(pattern_dir)

        pattern_path = osp.join(pattern_dir, f"ptn_{pre_sample_pattern_num}_{pattern_size}_{p}_{q}.pt")
        eid_path = osp.join(pattern_dir, f"eid_{pre_sample_pattern_num}_{pattern_size}_{p}_{q}.pt")
        if osp.exists(pattern_path) and osp.exists(eid_path):
            patterns = torch.load(pattern_path, map_location=torch.device('cpu'))
            eids = torch.load(eid_path, map_location=torch.device('cpu'))
            print('Done loading patterns from cache.')
        else:
            patterns, eids = get_patterns(graph, params)
            torch.save(patterns, pattern_path)
            torch.save(eids, eid_path)

        return {'pattern': patterns, 'eid': eids}


def train_node(graph, model, optimizer, split, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    bs = params['batch_size']

    total_loss, total_val_loss, total_test_loss = 0, 0, 0
    nodes = torch.arange(graph.num_nodes)
    y = graph.y
    if y.ndim == 2:
        y = y.squeeze()

    if split is not None:
        train_mask = split["train"]
    else:
        train_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    train_nodes = nodes[train_mask]
    train_num_nodes = train_nodes.size(0)
    train_num_batches = (train_num_nodes + bs - 1) // bs
    train_perm = torch.randperm(train_num_nodes)

    for i in range(train_num_batches):
        cur_nodes = train_nodes[train_perm[i * bs: (i + 1) * bs]]
        cur_y = y[cur_nodes].to(device)

        pred, instance_emb, pattern_emb, commit_loss = model(graph, cur_nodes, params, mode='train')
        if params.get('num_tasks') is not None:
            num_tasks = params['num_tasks']
            if num_tasks == 1:
                loss = F.binary_cross_entropy_with_logits(pred.squeeze(), cur_y.float())
            else:
                # loss = multitask_cross_entropy(pred, cur_y)
                loss = F.binary_cross_entropy_with_logits(pred, cur_y.float())
        else:
            loss = F.cross_entropy(pred, cur_y, label_smoothing=params['label_smoothing'])
        loss = loss + commit_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    total_loss /= train_num_batches
    total_val_loss /= train_num_batches
    total_test_loss /= train_num_batches

    return {'train': total_loss, 'val': total_val_loss, 'test': total_test_loss}


def eval_node(graph, model, split, params):
    model.eval()
    device = get_device_from_model(model)

    bs = params['batch_size']
    results = {'train': 0, 'metric': params['metric']}

    with torch.no_grad():
        for key in ['val', 'test']:
            mask = split[key]
            idx = mask2idx(mask)
            y = graph.y[idx].to(device)
            if y.ndim == 2:
                y = y.squeeze()

            num_batches = (len(idx) + bs - 1) // bs
            pred_list = []

            for i in range(num_batches):
                cur_nodes = idx[i * bs: (i + 1) * bs]
                pred, _, _, _ = model(graph, cur_nodes, params, mode='eval')
                pred_list.append(pred.detach())
            pred = torch.cat(pred_list, dim=0)

            results[key] = evaluate(pred, y, params=params)
    return results


def pretrain_node(graph, model, optimizer, scheduler=None, params=None):
    if params['inference_only']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)

    bs = params['batch_size']
    nodes = torch.arange(graph.num_nodes)
    if params['train_mask'] is not None:
        # if only use a subset of nodes for training
        nodes = nodes[params['train_mask']]
    num_nodes = nodes.size(0)
    nodes = nodes[torch.randperm(num_nodes)]
    num_batches = (num_nodes + bs - 1) // bs

    total_loss = 0
    for i in range(num_batches):
        cur_nodes = nodes[i * bs: (i + 1) * bs]
        loss = model.pretrain_node(graph, cur_nodes, params)
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


def linear_probe_node(graph, model, splits, params):
    """Linear probe evaluation for node classification"""
    model.eval()
    device = get_device_from_model(model)

    # Extract embeddings for all nodes
    bs = params['batch_size']
    nodes = torch.arange(graph.num_nodes)
    num_batches = (graph.num_nodes + bs - 1) // bs

    embeddings_list = []
    with torch.no_grad():
        for i in range(num_batches):
            cur_nodes = nodes[i * bs: (i + 1) * bs]
            _, instance_emb, _, _ = model(graph, cur_nodes, params, mode='eval')
            embeddings_list.append(instance_emb.detach().cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    y = graph.y
    if y.ndim == 2:
        y = y.squeeze()
        
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

        num_classes = params['output_dim']
        classifier = nn.Linear(embeddings.shape[1], num_classes).to(device)
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
                if params.get('num_tasks') is not None:
                    if params['num_tasks'] == 1:
                        loss = F.binary_cross_entropy_with_logits(batch_logits.squeeze(), batch_y.float())
                    else:
                        loss = F.binary_cross_entropy_with_logits(batch_logits, batch_y.float())
                else:
                    loss = F.cross_entropy(batch_logits, batch_y)

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
        best_val_accs.append(best_val_acc)
        best_test_accs.append(best_test_acc)

    return {'train': 0, 'train_std': 0, 'val': np.mean(best_val_accs), 'val_std': np.std(best_val_accs), 'test': np.mean(best_test_accs), 'test_std': np.std(best_test_accs), 'metric': params['metric']}
