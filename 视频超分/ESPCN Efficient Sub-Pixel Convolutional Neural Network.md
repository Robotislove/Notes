# ESPCN Efficient Sub-Pixel Convolutional Neural Network

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

本文核心：Efficient Sub-pixel Convolution。然而，虽然叫作Sub-pixel Convolution却并没有采用卷积操作。

## 整体架构

![image-20211104220234336](assets/image-20211104220234336.png)整个网络用了两个卷积层来提取特征，一个亚像素卷积层来整合特征。

**Input(1)->conv(5,64)->conv(3,32)->subpixel ->output(4)**

## 亚像素卷积层

几种上采样的方法：

1. Interpolation, 比如SRCNN就用简单的bicubic插值进行初步的上采样，然后进行学习非线性映射。

2. deconvolution, FSRCNN在最后的上采样层，通过学习最后的deconvolution layer。但deconvolution本质上是可以看做一种特殊的卷积，理论上后面要通过stack filters才能使得性能有更大的提升。 
3. 本文提到的亚像素卷积sub-pixel Layer，其实跟常规的卷积层没有什么不同，不同的是其输出的特征通道数为$r^2$，其中r为缩放倍数。 

$I^{SR} =f^L(I^{LR})=\mathcal PS(W_L*f_{L−1}(I^{LR})+b_L)$

其中$\mathcal PS$表示 **periodic shuffling**算子，将$H\times W\times C\cdot r^2$张量的元素重新排列成$rH\times rW\times C$形状的张量。

本质上就是将低分辨率特征，按照特定位置，周期性的插入到高分辨率图像中，可以通过颜色观测到上图的插入方式。

## 结果

<img src="assets/image-20211104222257305.png" alt="image-20211104222257305" style="zoom:50%;" />

速度上是相当快的，基本上能在视频上进行实时处理。

![image-20211104222420401](assets/image-20211104222420401.png)