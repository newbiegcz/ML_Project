# ML_Project

## 准备工作

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载 BTCV 数据集
将数据放到仓库根目录下，格式同 SwinUNETR。路径如下：
```bash
raw_data
├── dataset_0.json
├── imagesTr
│   ├── img0001.nii.gz
|   |   ...
│   └── img0040.nii.gz
├── labelsTr
│   ├── label0001.nii.gz
|   |   ...
│   └── label0040.nii.gz
├── ...
```

### 预处理数据集
```bash
python preprocess.py <embedding_disk_path> <datapoints_disk_path>
```

数据集将保存到这两个目录，文件较大，请尽可能保存到容量较大的 SSD 上。 (可能有 250G 左右)

### 准备 Checkpoint
将预训练 SAM 模型的权重文件 `sam_vit_h_4b8939.pth` 放到 `checkpoint` 目录下。

## Task 1
### 计算原始 SAM 模型的 mDice
```bash
python calc_pretrain_dice.py
```

## Task 2
### 训练
```bash
python train_task2.py fit --data.embedding_file_path <embedding_disk_path> --data.datapoint_file_path <datapoints_disk_path>
```

### 将 lightning checkpoint 转化为 pth 文件
lightning checkpoint 会保存在 `experiment_logs` 目录下。在用于后续的测试之前，需要先转化为 `pth` 文件。

```bash
python extract_task2.py <input_file> <output_file>
```

### 计算训练后模型的 mDice
```bash
python calc_finetune_dice.py <pth_path>
```

## Task 3
### 训练
```bash
python train_task3.py fit --data.embedding_file_path <embedding_disk_path> --data.datapoint_file_path <datapoints_disk_path>
```

### 将 lightning checkpoint 转化为 pth 文件
```bash
python extract_task3.py <input_file> <output_file>
```

### 在验证集上进行自动分割
```bash
python evaluate_task3.py [-h] [--checkpoint CHECKPOINT] [--file_key FILE_KEY] [--save_path SAVE_PATH] [--device DEVICE]
```

参数说明：
- `--checkpoint`：训练好的模型的 `pth` 文件路径
- `--file_key`：validation 或 training
- `--save_path`：分割结果保存路径
- `--device`：使用的设备，如 `cuda:0`


### 可视化自动分割结果
```bash
python visualize_result.py [-h] [--file_name FILE_NAME] [--save_path SAVE_PATH]
```

参数说明：
- `--file_name`：可视化的 CT 图文件名称，例如 `img0035`
- `--save_path`：上一步中，分割结果的保存路径
