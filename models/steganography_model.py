import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class stegGNN(nn.Module):
    def __init__(self,img_size,in_channels,embedding_dim)->None:
        super().__init__()
        self.encoder = Encoder(img_size,in_channels,embedding_dim)    
        self.decoder = Decoder(img_size,in_channels,embedding_dim)
        
    def encode(self,secret,cover):
        stego_image = self.encoder(secret,cover)
        return stego_image
    
    def decode(self,stego):
        reconstructed_secret_image = self.decoder(stego)
        return reconstructed_secret_image