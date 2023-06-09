from torch.utils.data import Dataset
import diskcache

class TrainDataset(Dataset):

    def __init__(self, *,
                    datapoint_file_path,
                    embedding_file_path,
                    model_type='vit_h'
                 ):   
        
        super().__init__()

        assert(model_type == 'vit_h'), "Only vit_h is supported for now."

        self.model_type = model_type

        self.embedding_cache = diskcache.Cache(embedding_file_path, eviction_policy = "none")
        self.datapoint_cache = diskcache.Cache(datapoint_file_path, eviction_policy = "none")

        self.num_image = self.embedding_cache["num_image_for_training"]
        self.num_datapoints = self.datapoint_cache["num_datapoints_for_training"]

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        res = dict()
        res["embedding"] = self.embedding_cache[("training", self.datapoint_cache[("training", idx)]["image_id"])]["embedding"]
        res["label"] = self.embedding_cache[("training", self.datapoint_cache[("training", idx)]["image_id"])]["label"]
        res["mask_cls"] = self.datapoint_cache[("training", idx)]["mask_cls"]
        res["prompt"] = self.datapoint_cache[("training", idx)]["prompt_point"]
        return res 

    def __del__(self):
        self.embedding_cache.close()
        self.datapoint_cache.close()

class ValidationDataset(Dataset):

    def __init__(self, *,
                    datapoint_file_path,
                    embedding_file_path,
                    model_type='vit_h'
                 ):   
        
        super().__init__()

        assert(model_type == 'vit_h'), "Only vit_h is supported for now."

        self.model_type = model_type

        self.embedding_cache = diskcache.Cache(embedding_file_path, eviction_policy = "none")
        self.datapoint_cache = diskcache.Cache(datapoint_file_path, eviction_policy = "none")

        self.num_image = self.embedding_cache["num_image_for_validation"]
        self.num_datapoints = self.datapoint_cache["num_datapoints_for_validation"]

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        res = dict()
        res["embedding"] = self.embedding_cache[("validation", self.datapoint_cache[("validation", idx)]["image_id"])]["embedding"]
        res["label"] = self.embedding_cache[("validation", self.datapoint_cache[("validation", idx)]["image_id"])]["label"]
        res["mask_cls"] = self.datapoint_cache[("validation", idx)]["mask_cls"]
        res["prompt"] = self.datapoint_cache[("validation", idx)]["prompt_point"]
        return res 

    def __del__(self):
        self.embedding_cache.close()
        self.datapoint_cache.close()