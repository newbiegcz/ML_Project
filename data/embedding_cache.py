import diskcache as dc
import torch

size_10gb = 3 * 1024 * 1024 * 1024 # 10 GB

class EmbeddingCache:
    def __init__(self, encoder, model_type, size_limit=size_10gb, path='embedding_cache', skip_check=False):
        self.encoder = encoder
        self.model_type = model_type
        self.size_limit = size_limit
        self.cache = dc.Cache(path, size_limit=size_limit)
        if not skip_check:
            assert model_type in ["vit_h", "vit_l", "vit_b"]
            if model_type == "vit_h":
                assert encoder.depth == 32, "vit_h encoder depth must be 32"
            elif model_type == "vit_l":
                assert encoder.depth == 24, "vit_l encoder depth must be 24"
            elif model_type == "vit_b":
                assert encoder.depth == 12, "vit_b encoder depth must be 12"

    def __del__(self):
        self.cache.close()

    def get_embedding(self, image: torch.Tensor):
        assert len(image.shape) == 3, "image must be 3d tensor"
        assert image.shape[0] == 3, "image must be in RGB format"
        image_key = image

        if self.cache.add(image_key, -1): # to avoid parallel access
            self.cache[image_key] = self.encoder(image)
        
        return self.cache[image_key]


    cache = dc.Cache('embedding_cache', size_limit=size_limit)
    # In [6]: cache = dc.Cache('tmp')
    # In [7]: cache[b'key'] = b'value'
    # In [8]: %timeit cache[b'key']

    cache.close()