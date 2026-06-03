# 数据

本仓库仅在 `data/samples/` 下保留少量示例图像和标签文件。

完整训练数据集未包含在仓库中，因为其文件体积较大，并且该数据集来源于 GHRSST/MUR-JPL Level 4 海表温度产品和人工标注结果。

预期的 YOLO 数据集目录结构如下：

```text
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

原始实验共使用 470 张已标注的海表温度锋图像块：

* 训练集：370 张图像
* 验证集：50 张图像
* 测试集：50 张图像

每个标签文件均遵循 YOLO 格式：

```text
class_id x_center y_center width height
```

本研究仅使用一个类别：`front`。
