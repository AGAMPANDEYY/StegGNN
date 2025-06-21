import torch

class Config:
    """Configuration class for steganography training"""
    # Training parameters
    num_epochs = 3000  
    initial_lr = 1e-4
    lr_decay_interval = 500
    lr_decay_factor = 0.5
    batch_size = 8
    
    # Loss weights
    revealing_weight = 0.75
    
    # Optimizer parameters
    optimizer_betas = (0.5, 0.999)
    weight_decay = 1e-5
    
    # Model parameters
    img_size = 256
    in_channels = 3
    embedding_dim = 192
    
    # Data paths
    train_dir = "/home/cvpr_ug_2/Stego1/steganography-div2k/dataset/train/clean"
    
    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
