# ML_Project

## 准备工作

### 安装依赖
```bash
pip install -r requirements.txt
```

### 下载 BTCV 数据集
格式同 SwinUNETR。将数据放到仓库根目录下，路径如下
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
python preprocess.py <datapoints_path> <embeddings_path>
```

数据集将保存到这两个目录。

## Task 1
### 计算原始 SAM 模型的 mDice
```bash
python calc_pretrain_dice.py
```



## Task 2
### 训练
```bash
python train_task2.py fit --data.embedding_file_path "/root/autodl-tmp/data_with_roi/embeddings" --data.datapoint_file_path "/root/autodl-tmp/data_with_roi/datapoints"
```

### 将 lightning checkpoint 转化为 pth 文件
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
python train_task3.py fit --data.embedding_file_path "/root/autodl-tmp/data_with_roi/embeddings" --data.datapoint_file_path "/root/autodl-tmp/data_with_roi/datapoints"
```

### 将 lightning checkpoint 转化为 pth 文件
```bash
python extract_task3.py <input_file> <output_file>
```

### 测试自动分割
```bash
python evaluate_task3.py [-h] [--checkpoint CHECKPOINT] [--file_key FILE_KEY] [--save_path SAVE_PATH] [--device DEVICE]
```

### 可视化自动分割结果
```bash
python visualize_result.py [-h] [--file_name FILE_NAME] [--save_path SAVE_PATH]
```