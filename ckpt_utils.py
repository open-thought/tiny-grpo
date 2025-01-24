from torch.distributed.checkpoint.state_dict import (
    set_optimizer_state_dict,
    set_model_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
)
import torch.distributed.checkpoint as dcp
import torch.distributed as dist

def save_checkpoint(model, optimizer, path):
    """Save model and optimizer state using distributed checkpoint"""
    model_state = get_model_state_dict(model=model)
    optimizer_state = get_optimizer_state_dict(model=model, optimizers=optimizer)
    
    state_dict = {'model': model_state,'optimizer': optimizer_state}
    
    dcp.save(state_dict=state_dict,storage_writer=dcp.FileSystemWriter(path))
    

def load_checkpoint(model, optimizer, path):
    """Load model and optimizer state using distributed checkpoint"""

    dcp_state_dict = {
        "model": get_model_state_dict(model=model),
        "optimizer": get_optimizer_state_dict(model=model, optimizers=optimizer),
    }
    
    dcp.load(dcp_state_dict, storage_reader=dcp.FileSystemReader(path))
    
    set_model_state_dict(model=model, model_state_dict=dcp_state_dict["model"])
    set_optimizer_state_dict(model=model, optimizers=optimizer, optim_state_dict=dcp_state_dict["optimizer"])

