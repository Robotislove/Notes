# MAE: Masked Autoencoders Are Scalable Vision Learners

GPT2：Language Models **are** Unsupervised Multitask **Learners**

GPT3：Language Models **are** Few-Shot **Learners**

FAIR,He Kaiming

## Abstract

证明了MAE是一种可扩展的自监督学习器。

MAE approach：mask random patches，然后重建。

1. 设计了一个非对称编码器-解码器体系结构，其中的编码器仅在可见的patches子集上运行（没有mask tokens）。解码器为轻量级解码器，从 latent representation 和mask tokens重建原始图像。
2. 发现 mask 输入图像比例很高的时候（例如75%），生成了一个 nontrivial and meaningful自监督学习任务。

这两种设计的结合使我们能够高效地训练大型模型：加快了训练速度3倍甚至更多，同时提高了准确性。

Scalable approach可以使得 **high-capacity** 模型泛化性很好：在仅使用 ImageNet-1K 数据的方法中，vanilla ViT-Huge 模型实现了最佳准确率 (87.8%)。

用 MAE 做 pre-training 只需 ImageNet-1k 就能达到超过 87% 的 top 1 准确度，超过了所有在 ImageNet-21k pre-training 的 ViT 变体模型。

下游任务中的迁移性能优于有监督的预训练，并显示出良好的scaling性能。

<img src="assets/image-20211119165403221.png" alt="image-20211119165403221" style="zoom:50%;" />

decoder的输入vector中，空白的（灰色的token）相当于是mask embedding，重建出所有patch，组合成目标。

重点：不对称的autoencoder，encoder非常大，decoder非常小，因为不需要decoder重建出图像（本来就知道图像）。

pre-training不是重点，不需要训练的非常好，重点是学习encoder的参数而不是decoder的参数。

问题：

1. 预训练时费时间
2. 引入pre-training和fine-tuning之间的gap。

masked autoencoders，一种更通用的去噪自编码器，在视觉上相关的研究[49,39]先于BERT。

BERT训练词向量的方式，就是将一个句子中的这个单词mask掉，然后根据这个词周围的其他词汇来预测这个mask掉的单词的语义；这种特征学习的方式在NLP中已经证明是work的了，但是在图像上一直没人去深入想一下为什么auto-encoder中间层的feature迁移来做分类效果很差。

为什么masked auto-encoding 在图像和文本的任务上差距如此之大？

1. **模型结构不同**；图像一般用的都是CNN，不好编码position embeddings这种位置相关的信息；对应的，现在有ViT这种transformer结构可以做到了；
2. **图像和文本的信息密度不一样**；文本的语义密度更大，而图像本身来说，却是有严重空间冗余的；对应的，可以通过随机mask掉图像中的一大部分patches来降低空间冗余，从而强制模型学到一些跨空间的语义上的特征；
3. **decoder重建是像素级别的任务**，相对于分类任务来说，学到的是一些low-level的feature，不足以用于语义相关的任务；虽然BERT中的decoder可以用一些简单的MLP来实现，但是对于图像来说，decoder的设计对于能学习到什么程度的语义来说非常重要；

同时，由于mask掉了一部分输入，可以让一些更大模型的训练变快了；

## Method

对方法中的一些细节进行了详细的说明和实验：

1. **Masking**，要用random masking mask掉很大一部分比例的patches。降低冗余，避免center bias；
2. **MAE encoder**，是一堆Transoformer Blocks，只encode了unmasked patches和position embeddings；
3. **MAE decoder**，也是一堆Transoformer Blocks，其输入包含了上一步encoded的visible patches和所有的 mask掉的 tokens两部分，另外，再加上position embeddings来让每一个tokens知道他们的位置应该在哪里；每一个mask token是一个共享的，可学习的vector；
4. **Reconstruction target**，只是用MSE Loss 重建了masked掉的patches；这是为了证明方法的有效性，要是想提高未mask掉区域的重建效果的话，也可以加上训；



