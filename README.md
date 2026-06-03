# 基于 YOLO11 的海洋温度锋检测方法研究

本项目使用 YOLO11n 从海表温度（Sea Surface Temperature，SST）图像中检测海洋温度锋。本项目整理自毕业设计“基于深度学习的海洋锋检测方法研究”。

原始实验使用了南海区域的 GHRSST Level 4 / MUR-JPL 海表温度数据。整体工作流程如下：

1. 读取 NetCDF 格式的 SST 数据。
2. 裁剪南海区域。
3. 将 SST 数据切片渲染为图像。
4. 使用 Sobel 梯度提取候选温度锋区域。
5. 对候选锋面进行人工筛选，并按照 YOLO 格式进行标注。
6. 训练并评估 YOLO11n 模型，用于温度锋检测。

## 仓库结构

```text
ocean-temperature-front-detection-yolo11/
  README.md
  requirements.txt
  .gitignore
  configs/
  data/
  src/
  results/
  weights/
  assets/
```

## 环境配置

```bash
pip install -r requirements.txt
```

如果本地没有 `yolo11n.pt` 文件，Ultralytics 可能会在训练开始时自动下载该模型权重。

## 数据

本仓库仅在 `data/samples/` 中包含少量示例图像和标签文件。

完整数据集应按照以下结构放置：

```text
data/images/train
data/images/val
data/images/test
data/labels/train
data/labels/val
data/labels/test
```

本实验共使用 470 张已标注的海表温度锋图像块：

* 训练集：370 张
* 验证集：50 张
* 测试集：50 张

类别定义如下：

```text
0: front
```

## 主要命令

查看 GHRSST 文件信息：

```bash
python src/read_ghrsst.py data/raw --max-files 3
```

裁剪南海区域：

```bash
python src/crop_south_china_sea.py --input-dir data/raw --output-dir data/interim/south_china_sea
```

生成 SST 图像：

```bash
python src/generate_sst_images.py --input-dir data/interim/south_china_sea --output-dir data/interim/sst_images
```

生成 Sobel 候选锋面掩膜：

```bash
python src/sobel_front_detection.py --input-dir data/interim/south_china_sea --output-dir data/interim/sobel_fronts
```

将二值掩膜转换为 YOLO 标签：

```bash
python src/generate_yolo_labels.py --mask-dir data/interim/sobel_fronts/masks --label-dir data/interim/yolo_labels
```

训练 YOLO11n：

```bash
python src/train_yolo11n.py --config configs/train_yolo11n.yaml
```

模型评估：

```bash
python src/evaluate.py --weights weights/best.pt --data configs/data.yaml
```

模型预测：

```bash
python src/predict.py --weights weights/best.pt --source data/samples/images
```

## 实验结果

主要训练设置如下：

* 模型：YOLO11n
* 训练轮数：30
* 图像尺寸：512
* batch size：4
* 优化器：AdamW
* 初始学习率：0.0006

最终训练曲线指标如下：

* precision：0.67516
* recall：0.70690
* mAP@0.5：0.72663
* mAP@0.5:0.95：0.33825

独立测试集评估指标如下：

* precision：0.62458
* recall：0.67692
* mAP@0.5：0.60208
* mAP@0.5:0.95：0.26980

更多图表结果可在 `results/` 和 `assets/` 中查看。

