import torch

def get_emb(angles):
    emb = torch.stack((angles.sin(), angles.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.channels_per_dim = channels // 3
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels_per_dim, 2).float() / self.channels_per_dim))
    
    def forward(self, points):
        '''
        :param points: A 2d tensor of size (batch_size, 3)
        '''
        emb_x = get_emb(points[..., 0] * self.inv_freq)
        emb_y = get_emb(points[..., 1] * self.inv_freq)
        emb_z = get_emb(points[..., 2] * self.inv_freq)
        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        # pad the last dimension of emb to self.channels
        if emb.shape[-1] < self.channels:
            assert self.channels - emb.shape[-1] < 3
            emb = torch.nn.functional.pad(emb, (0, self.channels - emb.shape[-1]))
        return emb

    
