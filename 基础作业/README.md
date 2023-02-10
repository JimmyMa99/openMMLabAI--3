## 基础实验
作业链接：
https://github.com/open-mmlab/OpenMMLabCamp/blob/main/AI%20%E5%AE%9E%E6%88%98%E8%90%A5%E5%9F%BA%E7%A1%80%E7%8F%AD/%E4%BD%9C%E4%B8%9A%E4%B8%89_mmsegmentation.md

# 使用MMSegmentation，在自己的数据集上，训练语义分割模型
1. 数据集标注（可选）

使用Labelme、LabelU等数据标注工具，标注多类别语义分割数据集，并保存为指定的格式。

2. 数据集整理

划分训练集、测试集

3. 使用MMSegmentation训练语义分割模型

在MMSegmentation中，指定预训练模型，配置config文件，修改类别数、学习率。

4. 用训练得到的模型预测

获得测试集图片或新图片的语义分割预测结果，对结果进行可视化和后处理。

5. 在测试集上评估算法的速度和精度性能

6. 使用MMDeploy部署语义分割模型（可选）

本课代码：https://github.com/TommyZihao/MMSegmentation_Tutorials/tree/main/20230206
checkpoints：链接：https://pan.baidu.com/s/1fv7_vgAS61QmlNZpz4xenA 提取码：88rz 

## 实验设备
NVIDIA Tesla P40 * 1

##  组织病理切片小鼠肾小球数据集

#### 数据集介绍


组织病理切片小鼠肾小球：https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/Glomeruli-dataset.zip



### 组织病理切片小鼠肾小球分割结果

|                |  IoU (%) |Acc(%)|
| :-----------------: |  :-------: | :-------: |
| background |   99.23   |99.76|
| glomeruili |   64.17   |71.98|

### 混淆矩阵
![图片](https://user-images.githubusercontent.com/101508488/218174224-7fcb452a-6c75-44b9-a390-9930c4f52eb7.png)

### 分割结果
![图片](https://user-images.githubusercontent.com/101508488/218174363-9640b132-eb75-4cb0-bb1d-40d517ebfa6d.png)
