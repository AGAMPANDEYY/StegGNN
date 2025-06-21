import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_layers import BasicConv

# Put all your graph-related functions and classes here:

def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)
        
def dense_knn_matrix(x, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        
        dist = pairwise_distance(x.detach())
           
        _, nn_idx = torch.topk(-dist, k=k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)
    
def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature
    
def window_partition(x, window_size=7):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    x = x.transpose(1,2).transpose(2,3)
    B, H, W, C = x.shape
    windows = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    windows = windows.transpose(2,3).transpose(1,2)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    windows = windows.transpose(1,2).transpose(2,3)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    x = x.transpose(2,3).transpose(1,2)
    return x

class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index
        
class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x):
        #### normalize
        x = F.normalize(x, p=2.0, dim=1)
        ####
        edge_index = dense_knn_matrix(x, self.k * self.dilation)
        return self._dilated(edge_index)
        
class GraphConvolution(nn.Module):
    """
    Static EdgeConv2d graph convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels])

    def forward(self, x, edge_index, y=None):
        
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value
        
class DyGraphConv2d(GraphConvolution):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1,
                 stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.debug = True


    def forward(self, x, relative_pos=None, adj_mask = None):
        # print('Doing gnn')
        B, C, H, W = x.shape

        x = x.reshape(B, C, -1, 1).contiguous()
                
        edge_index = self.dilated_knn_graph(x)

        x = super(DyGraphConv2d, self).forward(x, edge_index)
        

        if self.debug:
            return x.reshape(x.shape[0], -1, H, W).contiguous(), edge_index    
        return x.reshape(x.shape[0], -1, H, W).contiguous()
    
class WindowGrapher(nn.Module):
    """
    Local Grapher module with graph convolution and fc layers
    """
    def __init__(
            self,
            in_channels,
            kernel_size=9,
            windows_size = 8,
            dilation=1,
            stochastic=False,
            epsilon=0.0,
            shift_size = 4,
            r = 1,
            input_resolution = (64,64),
            adapt_knn = True
    ):
        super(WindowGrapher, self).__init__()

        if min(input_resolution) <= windows_size:
            # if window size is larger than input resolution, we don't partition windows
            shift_size = 0
            windows_size = min(input_resolution)
        assert 0 <= shift_size < windows_size, "shift_size must in 0-window_size"
       

        max_connection_allowed = (windows_size // r)**2
        if shift_size > 0:
            assert shift_size % r == 0
            max_connection_allowed = (shift_size // r)**2

        assert kernel_size <= max_connection_allowed, f'trying k = {kernel_size} while the max can be: {max_connection_allowed}'


        self.windows_size = windows_size
        self.shift_size = shift_size
        self.r = r

        n_nodes = self.windows_size * self.windows_size

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.graph_conv = DyGraphConv2d(in_channels, (in_channels * 2), kernel_size, dilation, stochastic, epsilon, r = r)
        
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        attn_mask = None
        adj_mask = None
        if self.shift_size > 0:
            print(f'Shifting windows!')
            H, W = input_resolution
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, 1, H, W))
            h_slices = (slice(0, -self.windows_size),
                        slice(-self.windows_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.windows_size),
                        slice(-self.windows_size, -self.shift_size),
                        slice(-self.shift_size, None))

            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w] = cnt
                    cnt += 1

            mask_windows_unf = window_partition(img_mask, self.windows_size)  # nW, 1, windows_size, windows_size,
            
            
            mask_windows = mask_windows_unf.view(-1, self.windows_size * self.windows_size)
        
            if self.r > 1:
                mask_windows_y = F.max_pool2d(mask_windows_unf, self.r, self.r)
                mask_windows_y = mask_windows_y.view(-1, (self.windows_size // self.r) * (self.windows_size // self.r))
            else:
                mask_windows_y = mask_windows
            
            attn_mask = mask_windows_y.unsqueeze(1) - mask_windows.unsqueeze(2) # nW x N x (N // r)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(1000000.0)).masked_fill(attn_mask == 0, float(0.0))

            # Get n_connections_allowed for each node in each windows
            if adapt_knn:
                print('Adapting knn!')
                adj_mask = torch.empty((attn_mask.shape[0], attn_mask.shape[1], kernel_size)) # nW x N x k
                for w in range(attn_mask.shape[0]):
                    for i in range(attn_mask.shape[1]):
                        all_connection = torch.sum(attn_mask[w,i] == 0)
                        scaled_knn = (kernel_size * all_connection) // (self.windows_size * (self.windows_size // r))
                        n_connections_allowed = int(max(scaled_knn, 3.0))
                        # print(f'Window: {w} node {i} - allowed_connection = {all_connection} (k = {n_connections_allowed})')
                        masked = torch.zeros(kernel_size - n_connections_allowed)
                        un_masked = torch.ones(n_connections_allowed)
                        adj_mask[w,i] = torch.cat([un_masked,masked],dim=0)

        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("adj_mask", adj_mask)

    def _merge_pos_attn(self, batch_size):
        if self.attn_mask is None:
            return None
            
        # Just repeat the attention mask for the batch size
        return self.attn_mask.repeat(batch_size, 1, 1)  # B, N, N
    
    def forward(self, x):


        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape

        # cyclic shift
        if self.shift_size > 0:
         
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        
        
        x = window_partition(x, window_size = self.windows_size)
        

        pos_att = self._merge_pos_attn(batch_size=B)
       
       
        x, edge_index = self.graph_conv(x, pos_att)

        
        x = window_reverse(x, self.windows_size, H=H, W=W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = self.fc2(x)

        x = x + _tmp
        
        return x
    