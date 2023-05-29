import diskcache
import torch
from torch.utils.data import Dataset
from torch.multiprocessing import Queue
import numpy as np
import torchvision
from collections import deque
import matplotlib.pyplot as plt
import random
import albumentations 
import albumentations.pytorch.transforms
from data.dataset import PreprocessForModel
from data.dataset import DictTransform
from modeling.build_sam import build_pretrained_encoder

size_10gb = 10 * 1024 * 1024 * 1024 # 10 GB

def get_image_key(image, model_type):
    # Warning: The image may have been normalized!!!

    # downsample the image to avoid floating point errors
    image = torchvision.transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)(image.unsqueeze(0)).squeeze(0)

    image_min = image.min()
    image_max = image.max()

    image = ((image - image_min) / (image_max - image_min) * 255).to(torch.uint8)

    return (model_type, image_min.item(), image_max.item(), image.numpy().tobytes())

all_cmaps = plt.colormaps()
exclude = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                      'turbo', 'nipy_spectral', 'gist_ncar'] + ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c']
exclude_ = []
for e in exclude:
    exclude_.append(e + "_r")
exclude += exclude_

class TorchSeed:
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        # backup random states
        self.old_random_state = random.getstate()
        self.old_np_rng_state = np.random.get_state()
        self.old_rng_state = torch.random.get_rng_state()

        # set random states
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        random.setstate(self.old_random_state)
        np.random.set_state(self.old_np_rng_state)
        torch.random.set_rng_state(self.old_rng_state)

def wrap_with_torchseed(func, seed=None):
    if seed is None:
        def wrapper(*args, **kwargs):
            with TorchSeed(kwargs['seed']):
                kwargs.pop('seed')
                return func(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            with TorchSeed(seed):
                return func(*args, **kwargs)
    return wrapper

class RandCmap:
    cmaps = [plt.get_cmap(c) for c in all_cmaps if c not in exclude]
    def __init__(self, use_torch=True):
        self.use_torch = use_torch
    def __call__(self, x, **kwargs):
        if len(x.shape) == 3:
            if self.use_torch:
                assert x.shape[0] == 1
                x = x[0]
            else :
                assert x.shape[2] == 1
                x = x[:, :, 0]
        y = self.cmaps[torch.randint(0, len(self.cmaps), (1,))](x)
        if self.use_torch:
            y = y[:, :, :3].transpose(2, 0, 1) # remove alpha channel and move channel to the front
            return torch.from_numpy(y)
        else:
            return y[:, :, :3]

def wrap_albumentations_transform(transform):
    def wrapper(x):
        res = transform(image=x['image'], mask=x['label'])
        return {
            'image': res['image'],
            'label': res['mask']
        }
    return wrapper

class Producer:
    prompt_per_mask=3
    def __init__(self, data_files, augment_data, queue: Queue, chunk_size, encoder_batch_size, encoder_device, model_type, cache, delay, seed, debug):
        self.augment_data = augment_data
        self.seed = seed
        self.data_files = data_files
        self.queue = queue
        self.chunk_size = chunk_size
        self.encoder_batch_size = encoder_batch_size
        self.encoder_device = encoder_device
        self.model_type = model_type
        self.delay = delay
        self.cache = cache
        self.debug = debug
        
        self.initialized = False
    
    def initialize(self):
        assert not self.initialized
        # 该函数在新的进程上执行
        # 在旧的进程上创建模型，会有 bug ! (可能是 cuda 上模型复制的问题)
        
        self.encoder = build_pretrained_encoder(self.model_type, True)
        
        self.encoder.to(self.encoder_device)
        
        from data.dataset import Dataset2D
        self.raw_dataset = Dataset2D(self.data_files, device=torch.device('cpu'), transform=None, dtype=np.float32)
        self.data_distribution = torch.zeros(len(self.raw_dataset), dtype=torch.float32)
        for i in range(0, len(self.raw_dataset)):
            self.data_distribution[i] = torch.bincount(self.raw_dataset[i]['label'].flatten().to(torch.int32)).count_nonzero().to(torch.float32)
            
        eps = 1. / 20
        
        self.data_distribution /= self.data_distribution.sum()
        self.data_distribution = (1 - eps) * self.data_distribution + eps * (1. / len(self.data_distribution)) # there will be eps probability that a image is uniformly sampled

        # TODO: 修改高度缩放
        # TODO: 修改采样 2D Image 的方式 (现在是每个高度都采样，冗余过多)

        # The elements in available_datapoints are tuples of (embedding, label, prompts, step)
        self.available_datapoint_sets = deque()
        self.current_step = 0

        self.seed_rng = torch.Generator(device='cpu')
        self.normal_rng = torch.Generator(device='cpu')
        self.seed_rng.manual_seed(self.seed)
        self.normal_rng.manual_seed(self.seed + 1)

        # 如果打算不使用水平切面，可能应该增加透视变换的 augmentation
        # TODO: color map 前先 jitter
        
        # Important: pass the random generator to the transforms to ensure reproducibility

        if self.augment_data:
            # data augmentation
            # TODO: RandCmap 可能太过 aggressive 了
            # TODO: 可以考虑使用 RandColorJitter
            def unsqueeze(x, **kwargs):
                return x.reshape(x.shape + (1,))
            
            def gen_clache(**kwargs):
                return albumentations.Compose([
                    albumentations.FromFloat(dtype="uint8"),
                    albumentations.CLAHE(**kwargs),
                    albumentations.ToFloat()
                ])
            # 在 RandCmap 前离散化可能不是明智的选择..
            self.transform_2d = (
                wrap_with_torchseed(
                    torchvision.transforms.Compose([
                        DictTransform(["image", "label"], lambda x : x.numpy()),
                        wrap_albumentations_transform(
                            albumentations.Compose([
                                albumentations.Lambda(image=unsqueeze, mask=unsqueeze),
                                albumentations.RandomBrightnessContrast(p=0.2),
                                albumentations.Lambda(image=RandCmap(use_torch=False)),
                                gen_clache(p=0.8),
                                albumentations.RandomBrightnessContrast(p=0.8),
                                albumentations.RandomGamma(p=0.8),
                                albumentations.HorizontalFlip(p=0.5),
                                albumentations.VerticalFlip(p=0.5),
                                albumentations.RandomRotate90(p=0.5),
                                albumentations.OneOf([
                                    albumentations.CropNonEmptyMaskIfExists(256, 256, p=1.),
                                ], p=.5),
                                albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)
                            ])
                        ),
                        PreprocessForModel(normalize=True),
                    ])
                )
            )
        else :
            self.transform_2d = wrap_with_torchseed (
                torchvision.transforms.Compose([
                    DictTransform(["image"], lambda x : x.expand(3, -1, -1)),
                    DictTransform(["label"], lambda x : x.unsqueeze(0)),
                    PreprocessForModel(normalize=True)
                ]))

        self.buffer_images = []
        self.buffer_image_keys = []
        self.buffer_labels = []

    def gen_datapoint_set(self, embedding, label, image=None):
        count = torch.bincount(label.flatten().to(torch.int32))
        prompts = []
        for i in range(0, len(count)):
            if count[i] >= self.prompt_per_mask:
                nonzero_indexes = torch.nonzero(label[0] == i)
                inds = torch.randint(len(nonzero_indexes), (self.prompt_per_mask,), generator=self.normal_rng)
                prompt = nonzero_indexes[inds]
                prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        prompts = prompts[torch.randperm(len(prompts), generator=self.normal_rng)]
        mask_cls = label[0, prompts[:, 0], prompts[:, 1]]
        if image is None:
            return [embedding, label, prompts, mask_cls, self.current_step]
        else :
            return [embedding, label, prompts, mask_cls, image, self.current_step]

    def gen_image(self):
        image_seed = torch.randint(1000000, (1,), generator=self.seed_rng).item()
        with TorchSeed(image_seed):
            # randomly select an image according to the data distribution
            image_index = torch.multinomial(self.data_distribution, 1).item()
            return self.transform_2d(self.raw_dataset[image_index], seed=image_seed+1)
        
    def process_buffer(self):
        images = torch.stack(self.buffer_images)
        if self.debug:
            print(self.encoder_device, "started", images.shape)
        with torch.inference_mode():
            _images = images.to(self.encoder_device).to(torch.float32)
            embeddings = self.encoder(_images).cpu()

        for i in range(len(embeddings)):
            embedding = embeddings[i]
            image_key = self.buffer_image_keys[i]
            label = self.buffer_labels[i]
            self.cache[image_key] = embedding
            # if self.debug:
            #     from training.sam_with_label.debug import initialize, get_image_embedding
            #     initialize(self.model_type)
            #     embedding_ = get_image_embedding(images[i], False, self.encoder)[0]
            #     assert embedding_.shape == embedding.shape
            #     # assert torch.abs(embedding - embedding_).max() < 1e-1
            #     embedding = embedding_
            datapoint_set = self.gen_datapoint_set(embedding, label, images[i].cpu() if self.debug else None)
            self.available_datapoint_sets.appendleft(datapoint_set)
        
        self.buffer_images = []
        self.buffer_image_keys = []
        self.buffer_labels = []
        if self.debug:
            print("end")

    def produce(self):
        if not self.initialized:
            self.initialize()

        while True:
            self.current_step += 1
            if len(self.available_datapoint_sets) == 0 or self.available_datapoint_sets[0][-1] > self.current_step:
                while len(self.buffer_images) < self.encoder_batch_size and (len(self.available_datapoint_sets) < self.chunk_size or self.available_datapoint_sets[0][-1] > self.current_step):
                    d = self.gen_image()
                    image = d['image']
                    label = d['label']
                    image_key = get_image_key(image, self.model_type)
                    if image_key in self.cache:
                        embedding = self.cache[image_key]
                        # if self.debug:
                        #     from training.sam_with_label.debug import initialize, get_image_embedding
                        #     initialize(self.model_type)
                        #     embedding_ = get_image_embedding(d['image'], False)[0]
                        #     assert embedding_.shape == embedding.shape
                        #     #assert torch.abs(embedding - embedding_).max() < 1e-1
                        #     embedding = embedding_
                            
                        datapoint_set = self.gen_datapoint_set(embedding, label, image if self.debug else None)
                        self.available_datapoint_sets.appendleft(datapoint_set)
                        if self.debug:
                            print("cache hit")
                        continue
                    else:
                        if self.debug:
                            print("cache miss")
                    self.buffer_images.append(image)
                    self.buffer_labels.append(label)
                    self.buffer_image_keys.append(image_key)

                if len(self.buffer_images) >= self.encoder_batch_size:
                    self.process_buffer()

            assert self.available_datapoint_sets[0][-1] <= self.current_step

            datapoint_set = self.available_datapoint_sets.popleft()
            datapoint = {
                "embedding": datapoint_set[0],
                "label": datapoint_set[1],
                "prompt": datapoint_set[2][-1],
                "mask_cls": datapoint_set[3][-1],
            }
            datapoint['prompt'] = datapoint['prompt'][[1, 0]] # to make it compatible with the model
            if self.debug:
                # print ("normalized pixel max, min:", torch.max(datapoint_set[4]), torch.min(datapoint_set[4]))
                datapoint['image'] = (datapoint_set[4]  * PreprocessForModel.pixel_std + PreprocessForModel.pixel_mean)
                # print (datapoint['image'])
            self.queue.put(datapoint)
            datapoint_set[-1] += self.delay
            datapoint_set[2] = datapoint_set[2][:-1]
            datapoint_set[3] = datapoint_set[3][:-1]
            if len(datapoint_set[2]) > 0:
                self.available_datapoint_sets.append(datapoint_set)

class ExpDataset(Dataset):
    '''
    This is a hack to make an iterable dataset work with the dataloader. The __getitem__ function returns a different value every time it is called.

    The data is generated on the fly, and the embedding is cached to disk.

    When being used with torch.utils.data.DataLoader, the number of workers should be set to 0.

    If this is used with DistributedDataParallel, the seed should be set according to the rank of the process to avoid duplicated data.

    The order of data is not guaranteed to be consistent across runs, as it can be affected by whether the data is cached or not.
    However, the set of data is guaranteed to be the same across runs. It will only be different if the seed is changed.
    '''

    # TODO: Support multiple encoders
    def __init__(self, *,
                    data_files,
                    augment_data,
                    epoch_len,
                    chunk_size, 
                    encoder_batch_size, 
                    model_type='vit_h',
                    size_limit=size_10gb,
                    path='embedding_cache',
                    encoder_device=torch.device('cpu'),
                    delay=100,
                    debug=False,
                    seed
                 ):   
        '''
        Args:
            augment_data: whether to augment the data
            epoch_len: the number of data points in an epoch
            chunk_size: the number of data points to be generated at a time
            encoder_batch_size: the batch size to be used when calculating the embedding
            model_type: the model type of the encoder. should be one of "vit_h", "vit_l", "vit_b". This is used when cacheing the embedding. Default to "vit_h".
            size_limit: the size limit of the cache. Default to 10 GB.
            path: the path to the cache. Default to "embedding_cache".
            encoder_device: the device to be used when calculating the embedding. Default to cpu.
            delay: the minimum number of steps before the same image can be used again. Default to 100.
            seed: the seed to be used when generating the data. Important: this should be set according to the rank of the process when using DistributedDataParallel, or else the data will be duplicated.
        '''
        self.model_type = model_type
        self.size_limit = size_limit

        self.cache = diskcache.Cache(path, size_limit=size_limit)

        self.queue = Queue(maxsize=chunk_size)
        self.producer = Producer( 
                            data_files=data_files,  
                            queue=self.queue, 
                            chunk_size=chunk_size,
                            encoder_batch_size=encoder_batch_size,
                            encoder_device=encoder_device,
                            model_type=model_type,
                            cache=self.cache,
                            seed=seed,
                            delay=delay,
                            debug=debug,
                            augment_data=augment_data,
                        )
        self.process = torch.multiprocessing.Process(target=self.producer.produce)
        self.epoch_len = epoch_len
        self.process.start()

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        return self.queue.get()

    def __del__(self):
        if torch.utils.data.get_worker_info() is None:
            self.process.terminate()
            self.cache.close()

    def __getstate__(self):
        '''
        This is to make the dataset pickleable so that you can spawn multiple processes to load the dataset.
        Or else, you will get an error like this:
        ```
        TypeError: cannot pickle 'weakref' object
        ```
        '''
        state = self.__dict__.copy()
        state['process'] = None
        return state

def main():
    from modeling.build_sam import build_sam_with_label_vit_h
    from third_party.segment_anything.build_sam import build_sam_vit_h
    def get_sam_with_label(pretrained_checkpoint, build_encoder=True):
        org_sam = build_sam_vit_h(pretrained_checkpoint)
        org_state_dict = org_sam.state_dict()
        sam_with_label, encoder_builder = build_sam_with_label_vit_h(None, build_encoder=build_encoder)
        new_state_dict = sam_with_label.state_dict()
        for k, v in org_state_dict.items():
            if not build_encoder and k.startswith("image_encoder"): 
                continue  # sam_with_label has no image_encoder when build_encoder=False.
            assert k in new_state_dict.keys()
            new_state_dict[k] = v
        sam_with_label.load_state_dict(new_state_dict)
        return sam_with_label, encoder_builder
    checkpoint_path = "checkpoint/sam_vit_h_4b8939.pth"
    sam, encoder_builder = get_sam_with_label(checkpoint_path, False)
    from data.dataset import data_files
    dataset = ExpDataset(
        data_files=data_files["training"][:1],
        epoch_len=100,
        chunk_size=10,
        encoder_builder=encoder_builder,
        encoder_device=torch.device('cuda:0'),
        encoder_batch_size=2,
        model_type='vit_h',
        size_limit=size_10gb,
        path='embedding_cache',
        seed=1166117,
        delay=100,
        debug=True,
        augment_data=True,
    )

    # TODO: Test batch size > 1
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)

    import utils.visualize as viz
    
    viz.initialize_window()

    i = 0
    for data in dataloader:
        i += 1
        prompt_point = data['prompt'][0]
        prompt = [(prompt_point, 0)]
        dummy_image = torch.zeros((3, 1024, 1024))
        cls = data['label'][0][0][prompt_point[1]][prompt_point[0]]
        assert(cls == data['mask_cls'])
        cls = cls.to(torch.int32)
        viz.add_object_2d("image" + str(i),
                          image=data['image'][0].numpy(),
                          pd_label=None,
                          gt_label=data['label'][0].numpy(),
                          prompt_points=prompt,
                          label_name=viz.default_label_names,
                             extras={
                              "prompt": prompt_point,
                              "prompt_label": viz.default_label_names[cls]
                          }
        )

if __name__ == "__main__":
    main()