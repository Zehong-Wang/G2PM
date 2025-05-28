import psutil
import resource
import torch


def set_memory_limit(PERCENTAGE_MEMORY_ALLOWED=0.99):
    num_gpus = torch.cuda.device_count()
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available
    soft = int(available_memory * PERCENTAGE_MEMORY_ALLOWED / num_gpus)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f'Per GPU (Number of GPUs: {num_gpus}) Soft: {soft / 1024 / 1024 / 1024:.2f}G, Hard: {available_memory / 1024 / 1024 / 1024:.2f}G')
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
