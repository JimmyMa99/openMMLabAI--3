## 进阶实验
我们鼓励同学们使用mmsegmentation完成自己的项目，为此我们提供以下数据集给同学们使用，也欢迎同学们使用自己的数据集基于mmsegmentation做项目。完成3次进阶作业的同学可以获得额外积分激励。

👁️语义分割数据集：
https://opendatalab.org.cn/PASCAL_VOC2007
https://opendatalab.org.cn/PASCAL_VOC2012

## 实验设备
NVIDIA GeForce RTX 3090 *2

## 实验设计
以弱监督伪标签作为监督，使用mmseg中deeplabV3+作为WSSS的第二阶段进行训练。

## 数据集 Pascal VOC 2012 伪标签
使用图像级标签生成伪标签，使用伪标签训练分割网络
![图片](https://user-images.githubusercontent.com/101508488/218186496-ab30f197-141e-4e67-8dbe-ba4eca1f5383.png)


|        Model        |  mIoU (%) |
| :-----------------: |  :-------: |
|DeeplabV3+(ResNet101)|   70   |

 checkpoints：链接：https://pan.baidu.com/s/1uvgWQXtPGAFHYGVjm7g1cg 提取码：bzpg 
