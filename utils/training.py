import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by a factor every config.lr_decay_interval epochs"""
    lr = config.initial_lr * (config.lr_decay_factor ** (epoch // config.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def forward_pass(secret_img, cover_img, model, criterion, config):
    """Perform forward pass"""
    stego_img = model.encode(secret_img, cover_img)
    reconstructed_secret_img = model.decode(stego_img)
        
    hiding_error = criterion(cover_img, stego_img)
    revealing_error = criterion(secret_img, reconstructed_secret_img)
    loss = hiding_error + config.revealing_weight * revealing_error
    
    return loss, hiding_error, revealing_error

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_start = checkpoint["epoch"]
    return epoch_start, checkpoint["loss"]
