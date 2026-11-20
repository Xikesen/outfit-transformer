import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows, torch.distributed only supports the Gloo backend
# with FileStore and TcpStore. For FileStore, set the init_method
# argument of init_process_group to a local file, e.g.:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# TcpStore works the same way as on Linux.

def setup(
    rank: int, world_size: int
):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ.setdefault('MASTER_PORT', '12355')  # override to run two jobs at once

    # Initialize the process group
    dist.init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()
    
    
def gather_results(all_loss, all_preds, all_labels):
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return all_loss, all_preds, all_labels
    
    gathered_preds = [torch.empty_like(all_preds) for _ in range(dist.get_world_size())]
    gathered_labels = [torch.empty_like(all_labels) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_labels, all_labels)
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
    all_loss /= dist.get_world_size()
    gathered_preds = torch.cat(gathered_preds, dim=0)
    gathered_labels = torch.cat(gathered_labels, dim=0)
    
    return all_loss, gathered_preds, gathered_labels