import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR


def get_scheduler(optimizer, params):
    if params['scheduler'] == 'none':
        scheduler = None
    elif params['scheduler'] == 'inv_sqrt':
        scheduler = get_inverse_sqrt_scheduler(optimizer, params)
    elif params['scheduler'] == 'cosine':
        scheduler = get_cosine_with_warmup_scheduler(optimizer, params)
    else:
        raise NotImplementedError("The scheduler is not implemented.")

    return scheduler



def get_inverse_sqrt_scheduler(optimizer, params):
    warmup_epochs = params['warmup_epochs']
    warmup_steps = warmup_epochs * params['steps_per_epoch']
    d_model = params['hidden_dim']

    def lr_lambda(step):
        if step == 0:
            return 0.0
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_with_warmup_scheduler(optimizer, params):
    warmup_epochs = params['warmup_epochs']
    warmup_steps = warmup_epochs * params['steps_per_epoch']
    num_training_steps = params['epochs'] * params['steps_per_epoch']
    min_lr = params['min_lr']
    base_lr = params['lr']
    warmup_lr = params['warmup_lr']

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return (warmup_lr + (base_lr - warmup_lr) * (step / warmup_steps)) / base_lr
        else:
            # Cosine decay
            decay_ratio = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return (min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_ratio))) / base_lr

    return LambdaLR(optimizer, lr_lambda)