# STVSR 时域-空域同时超分

Temporal Modulation Network for Controllable Space-Time Video Super-Resolution

南开、中科院自动化所模式识别国家实验室、腾讯优图

## 摘要

低分辨率-低帧率视频超分。

可变形卷积，只能推断出训练阶段预定义的中间帧。

同时低估了相邻帧之间的短期运动。undervalued the short-term motion cues among adjacent frames

1. 提出TMNet（Temporal Modulation Network），用高分辨率重建插值任意数量中间帧。
2. 提出TMB（Temporal Modulation Block），调节可变形卷积核以实现可控的特征插值。

为了更好地利用时间信息，提出

1. LFC（Locally-temporal Feature Comparison），局部时间特征比较模块，视频中的短期运动信息
2. Globally-temporal feature fusion，实际是双向可变形ConvLSTM，视频中的长期运动信息。

## 一、简介

![](assets/20211129_220255_image.png)

### 1.1 Motivation

LCD（liquid-crystal display）液晶显示器、LED发光二极管（light-emitting diode）可以显示超高清视频（UHD TV）Ultra High Definition Television。

- 分辨率4K (3840 × 2160) or 8K (7680 × 4320)全彩像素

* 帧率：120fps甚至240fps

然而，当前视频仅仅FHD，2K (1920 × 1080)，30fps。为了将FHD在UHD TV上播放，需要增加时域和空域的分辨率。

虽然可以用图像超分逐帧提高视频的空间分辨率，但超分视频的感知质量会在时域上因时间失真而越来越差。因此提出STVSR同时进行时域和空域的超分。

过去的方法非常依赖精准的时域空域配准，配准不准确的话，效果不好。同时计算需求大，推理效率低。

后来DNN引入，但是一开始他们依次进行VFI和VSR，这样会忽略时间和空间维度之间的内在关系。

高分辨率帧在移动的对象和背景上有着丰富的细节，高帧率帧在相邻帧之间有更精准的像素对齐。

因此之前的STVSR会有temporal inconsistency problem，时间不一致问题，造成伪影。“the attentional blink phenomenon”

后来提出了许多一步STVSR方法，同时进行VSR和VFI

* STARnet用一个额外的光流分支做运动估计，对两个相邻帧做特征扭曲，然后插值中间帧。光流估计大量的算力和内存。
* Zooming slow-mo，不用光流，直接用可变形卷积作为backbone，并且在特征空间直接STVSR。

目前的STVSR效果还不错，但是只能生成网络结构预先定义的中间帧（30fps->60fps），所以只能用于固定帧率视频的高度受限的应用场景。 然而许多商业场景（比如体育），需要灵活地调整中间视频帧，以获得更好的可视化效果。 因此，需要**可控STVSR方法，smooth motion运动合成。**

### 1.2 效果

TMNet可以生成任意数量的中间帧， TMB将运动信息合并到中间帧的特征插值中，由时间参数定义的任意时刻的，可控插值。

此外，还有LFC局部特征比较模块融合多帧特征，用于有效的空间对齐和特征扭曲。 globally-temporal feature fusion探索整个视频的长期变化。 通过这两步-时域特征融合方法，精准插帧。

实验表明，TMNet可以插值任意数量的中间帧。STVSR上SoTA性能。

### 1.3 Contributions

1. 提出TMNet可控任意数量的插帧。主要是可变形卷积框架下的TMB实现的。
2. 两步时域特征统合，保证准确性和效率。
3. 实验SoTA

## 二、相关工作

### 2.1 VFI

早期主要是光流，

* Jiang，用光流做了任意帧率的VFI运动motion interpretation建模
* Niklaus，用上下文信息扭曲输入帧，插入上下文感知的中间帧。
* Bao，Memc-net，做运动估计和运动补偿。
* Niklaus，通过softmax splatting，解决了在VFI中将多个像素映射到同一位置的冲突。

然而，光流做运动估计的问题是，巨大的计算开销。

因此，最近提出，空间自适应卷积核，可变形卷积核做VFI。

### 2.2 VSR

主要是光流，aggregate空间信息。

* Jo，DUF，dynamic upsampling filters，残差网络
* Wang，EDVR，PCD模块用于帧对齐，通过空间和时间注意将多个帧融合为一个帧。
* Haris，Recurrent back-projection，迭代细化，集成多个帧的空间和时间上下文。
* Tian，TDAN，可变形卷积，offset，帧对齐。

### 2.3 STVSR

* Shechtman，directional space-time smoothness regularization
* Mudenagudi，马尔科夫随机场
* STARnet，额外光流分支，利用时域和空域固有的运动信息，对两个相邻帧扭曲，插值中间帧
* Xiang，Zooming slow-mo，PCD对齐模块插值多帧特征，Bi-directional可变形ConvLSTM来插值中间帧特征，最后多帧特征融合。

TMNet的效果更好，因为局部时间特征比较模块LFC。

### 2.4 Modulation networks

通过一个额外的调节分支，来控制主网络的恢复强度。 

本质是restortion quality和flexibility之间的trade-off。



## 三、方法

## 四、实验

### 4.1 数据集

### 4.2 模型设置

### 4.3 实验结果

### 4.4 消融实验

## 五、总结

## 六、思考
