import torch

def get_emb(angles):
    emb = torch.stack((angles.sin(), angles.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.channels_per_dim = channels // 6 * 2
        scale = 30.0
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, self.channels_per_dim, 2).float() / self.channels_per_dim)) * scale)
    
    def forward(self, points):
        '''
        :param points: A 2d tensor of size (batch_size, 3)
        '''
        emb_x = get_emb(points[:, 0].unsqueeze(1) * self.inv_freq.unsqueeze(0))
        emb_y = get_emb(points[:, 1].unsqueeze(1) * self.inv_freq.unsqueeze(0))
        emb_z = get_emb(points[:, 2].unsqueeze(1) * self.inv_freq.unsqueeze(0))
        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        ret = torch.zeros((points.shape[0], self.channels), dtype=points.dtype, device=points.device)
        ret[:, :emb.shape[1]] = emb
        return ret
