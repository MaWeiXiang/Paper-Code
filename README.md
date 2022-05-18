# Paper&Code
Record my research and encourage myself

     更新：2022/4/28
     更新：2022/4/4
     创建：2022/3/28
## Content
### [DRL-paper](https://github.com/MaWeiXiang/Paper-Code/edit/main/README.md#drl-paper-1)
### [ML-paper](https://github.com/MaWeiXiang/Paper-Code/edit/main/README.md#ml-paper-1)

------

### DRL-paper
* [2018 人工智能技术在安全漏洞领域的应用](http://www.infocomm-journal.com/txxb/CN/article/downloadArticleFile.do?attachType=PDF&id=167583)
> * 如何将机器学习、自然语言处理等人工智能技 术应用于安全漏洞的研究已成为新的热点
> * 首先分析了安全漏洞的**自动化挖掘、自动化评估、自动化利用和自动化修补**等关键技术，指出安全漏洞挖掘的自动化是人工智能在安全漏洞领域应用的重点，然后分析和归纳了近年来提出的将人工智能技术应用于安全漏洞研究的最新研究成果，指出了应用中的一些问题，给出了相应的解决方案。

### ML-paper
* （）On the Effectiveness of Local Binary Patterns in Face,
* (IEEE 2012) A face antispoofing database with diverse attacks,【[Paper](http://www.cbsr.ia.ac.cn/users/zlei/papers/ICB2012/ZHANG-ICB2012.pdf)】

* (arXiv 2022) Nighborhood Attention Transformer,【[Paper](https://arxiv.org/abs/2204.07143)】【[Code](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)】

##### 【AI DIGEST】
      **What this paper is about
      It is different from self attention being applied to local windows, and can be thought of as a convolution with a content-dependant kernel.
      We compare this module in terms of complexity and memory usage to self attention, window self attention, and convolutions.
      Both window self attention and neighborhood attention have a linear computational cost and memory usage with respect to resolution, as opposed to self attention's quadratic cost and memory.
      **What you can learn
      Through the use of overlapping convolutions, and our NAT design boosts, classification accuracy is significantly while resulting in fewer parameters and FLOPS than Swin-T. Swapping SWSA with NA results in an improvement of almost 0.5% in accuracy.
      In this paper, we introduced an alternate way of localizing self attention with respect to the structure of data, which computes key-value pairs dynamically for each token, along with a more data efficient configuration of models. This helps create a model that utilizes both the power of attention, as well as the efficiency and inductive biases of convolutions.
      We've shown the power of such a model in image classification, in which it outperforms both Swin Transformer and ConvNeXt significantly.

* (CVPR 2022) MPViT Multi-Path Vision Transformer for Dense Prediction,【[Paper](https://arxiv.org/abs/2112.11010)】【[Code](https://github.com/youngwanLEE/MPViT)】

##### 【AI DIGEST】
      **What this paper is about
      It is crucial for dense prediction tasks such as object detection and segmentation to represent features at multiple scales for discriminating between objects or regions of varying sizes.
      In this work, we focus on how to effectively represent multi-scale features with Vision Transformers for dense prediction tasks.
      Following the standard training recipe as in DeiT, we train MPViTs on ImageNet-1K, which consistently achieve superior performance compared to recent SOTA Vision Transformers.
      **What you can learn
      To validate the effectiveness of our attention map qualitatively, we compare attention maps of MPViT and CoaT.
      The attention maps of path-2 showcase the changing behavior between paths-1 and 3 since the scale of path-2 is in-between the scales of paths-1 and 3, and accordingly, the attention maps also begin to transition from smaller to larger objects.
      In other words, although the attention map of path-2 attends similar regions as path-1, it is also more likely to emphasize larger objects, as path-3 does.
      We test all models on the same Nvidia V100 GPU with a batch size of 256.
      Lite Small with a simple multi-stage structure similar to Swin.
      Lite and path-1 have similar patch sizes and show similar attention maps.
      Lite simultaneously attend to large and small objects, as shown in the 4th row.
      Lite on classification, detection, and segmentation tasks.
      The input image and corresponding attention maps from each path are illustrated.

* (CVPR 2022) MetaFormer is Actually What You Need for Vision,【[Paper](https://arxiv.org/abs/2111.11418)】

##### 【AI DIGEST】
      **What this paper is about
      By regarding the attention module as a specific token mixer, we further abstract the overall transformer into a general architecture MetaFormer where the token mixer is not specified, as shown in.
      To verify this hypothesis, we apply an extremely simple non-parametric operator, pooling, as the token mixer to conduct the most basic token mixing.
      Specifically, by only employing a simple non-parametric operator, pooling, as an extremely weak token mixer for MetaFormer, we build a simple model named PoolFormer and find it can still achieve highly competitive performance.
      **What you can learn
      Thus, we replace the token mixer pooling with attention or spatial FC 1 in the top one or two stages in PoolFormer.
      It achieves 81.0% accuracy with only 16.5M parameters and 2.7G MACs.
      In this work, we abstracted the attention in transformers as a token mixer, and the overall transformer as a general architecture termed MetaFormer where the token mixer is not specified.

* (NIPS 2021) XCiT Cross-Covariance Image Transformers,【[Paper](https://arxiv.org/abs/2106.09681)】

##### 【AI DIGEST】
      **What this paper is about
      Self-attention, as introduced by Vaswani et al., operates on an input matrix X 2 R N d where N is the number of tokens, each of dimensionality d. The input X is linearly projected to queries, keys and values, using the weight matrices W q 2 R ddq W k 2 R dd k and W v 2 R ddv such that Q=XW q K=XW k and V =XW v where d q = d k. Keys and values are used to compute an attention map A. Softmax, and the output of the self-attention operation is defined as the weighted sum of N token features in V with the weights corresponding to the attention map.
      The non-zero part of the eigenspectrum of the Gram and covariance matrix are equivalent, and the eigenvectors of C and G can be computed in terms of each other.
      We draw upon this strong connection between the Gram and covariance matrices to consider whether it is possible to avoid the quadratic cost to compute the N N attention matrix, which is computed from the analogue of the N N Gram matrix QK > =XW q W > k X.
      **What you can learn
      XCiT with 1616 patches provides a strong performance especially for smaller models where XCiT-S12/16 outperforms Swin-T. We present an alternative to token self-attention which operates on the feature dimension, eliminating the need for expensive computation of quadratic attention maps.
      In particular, it exhibits a strong image classification performance on par with state-of-the-art transformer models while similarly robust to changing image resolutions as convnets.
      We use report the results provided by the authors in their open-sourced code https://github.com.


* (arXiv 2021) Pyramid Vision Transformer A Versatile Backbone for Dense Prediction,【[Paper](https://arxiv.org/abs/2102.12122)】

##### 【AI DIGEST】
      **What this paper is about
      Performance comparison on COCO val2017 of different backbones using RetinaNet for object detection, where "T", "S", "M" and "L" denote our PVT models with tiny, small, medium and large size.
      Specifically, as illustrated in, our PVT overcomes the difficulties of the conventional Transformer by taking fine-grained image patches as input to learn high-resolution representation, which is essential for dense prediction tasks; introducing a progressive shrinking pyramid to reduce the sequence length of Transformer as the network deepens, significantly reducing the computational cost, and adopting a spatial-reduction attention layer to further reduce the resource consumption when learning high-resolution features.
      We propose Pyramid Vision Transformer, which is the first pure Transformer backbone designed for various pixel-level dense prediction tasks.
      **What you can learn
      Similar to the traditional Transformer, the length of ViT's output sequence is the same as the input, which means that the output of ViT is singlescale.
      Benefiting from the above designs, our method has the following advantages over ViT: 1 more flexible-can generate feature maps of different scales/channels in different stages; 2 more versatile-can be easily plugged and played in most downstream task models; 3 more friendly to computation/memory-can handle higher resolution feature maps or longer sequences.
      To provide instances for discussion, we design a series of PVT models with different scales, namely PVT.
      In, we also present some qualitative object detection and instance segmentation results on COCO val2017, and semantic segmentation results on ADE20K. These results indicate that a pure Transformer backbone without convolutions can also be easily plugged in dense prediction models, and obtain high-quality results.
      We introduce PVT, a pure Transformer backbone for dense prediction tasks, such as object detection and seman.
      Extensive experiments on object detection and semantic segmentation benchmarks verify that our PVT is stronger than well-designed CNN backbones under comparable numbers of parameters.

* (ICCV 2021) Multiscale Vision Transformers,【[Paper](https://arxiv.org/abs/2104.11227)】

##### 【AI DIGEST】
      **What this paper is about
      We present Multiscale Vision Transformers, a transformer architecture for modeling visual data such as images and videos.
      Our focus in this paper is video recognition, and we design and evaluate MViT for video tasks.
      We further apply MViT to ImageNet classification, by simply removing the temporal dimension of the video architecture, and show significant gains over single-scale vision transformers for image recognition.
      **What you can learn
      We further train a deeper 24-layer model with longer sampling, MViT-B-24, 323, to investigate model scale on this larger dataset.
      B is further improved by increasing the number of input frames and MViT.
      B layers and using K600 pre-trained models.
      In, we analyze the speed/accuracy trade-off of our MViT models, along with their counterparts vision transformer and Con-vNets.
      We have presented Multiscale Vision Transformers that aim to connect the fundamental concept of multiscale feature hierarchies with the transformer model.
      In empirical evaluation, MViT shows a fundamental advantage over single-scale vision transformers for video and image recognition.

* (arXiv 2021) Local-to-Global Self-Attention in Vision Transformers,【[Paper](https://arxiv.org/abs/2107.04735)】

##### 【AI DIGEST】
    **What this paper is about
    In a CNN, convolutional kernels are locally connected to the input feature maps, where features only interact with their local neighbors.
    In this work, we study the necessity of global and local feature interactions in self-attention modules of Vision Transformers and propose a local-to-global multi-path mechanism in self-attention, which we refer to as the LG.
    Transformer achieves competitive results on both image classification and semantic segmentation.
    **What you can learn
    In this section, we report the experimental results of our LG.
    Transformer on two computer vision tasks, i.e., image classification, and semantic segmentation.
    Experiments on image classification are conducted on the ImageNet-1K dataset which contains 1,000 classes, and 1.28M and 50K images for the training set and validation set, respectively.
    From this perspective, our model achieves a good trade-off between performance and efficiency.
    By building a multi-path structure in the hierarchical Vision Transformer framework, our model takes advantage of both local the feature learning mechanisms in CNNs and global feature learning mechanisms in Transformers.
    We conduct thorough studies on two computer vision tasks, and the results demonstrate that our framework yields improved performance with limited sacrifice in model parameters and computational overhead.

* (NIPS 2021) Focal Self-attention for Local-Global Interactions in Vision Transformers,【[Paper](https://arxiv.org/abs/2107.00641)】【[Code](https://github.com/microsoft/Focal-Transformer)】

##### 【AI DIGEST】
    **What this paper is about
    In this paper, we present a new self-attention mechanism to capture both local and global interactions in Transformer layers for high-resolution inputs.
    Based on the proposed focal self-attention, a series of Focal Transformer models are developed, by 1 exploiting a multi-scale architecture to maintain a reasonable computational cost for high-resolution images, and 2 splitting the feature map into multiple windows in which tokens share the same surroundings, instead of performing focal self-attention for each token.
    We validate the effectiveness of the proposed focal self-attention via a comprehensive empirical study on image classification, object detection and segmentation.
    **What you can learn
    Considering our focal attention prompts local and global interactions at each Transformer layer, one question is that whether it needs less number of layers to obtain similar modeling capacity as those without global interactions.
    More importantly, using two less layers, our model achieves comparable performance to Swin Transformer.
    Though extensive experimental results showed that our focal selfattention can significantly boost the performance on both image classification and dense prediction tasks, it does introduce extra computational and memory cost, since each query token needs to attend the coarsened global tokens in addition to the local tokens.

* (ICCV 2021) CvT Introducing Convolutions to Vision Transformers,【[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html)】【[Code](https://github.com/microsoft/CvT)】

##### 【AI DIGEST】

    ** What this paper is about
    Transformers have recently dominated a wide range of tasks in natural language processing.
    The beginning of each stage consists of a convolutional token embedding that performs an overlapping convolution operation with stride on a 2D-reshaped token map, followed by layer normalization. This allows the model to not only capture local information, but also progressively decrease the sequence length while simultaneously increasing the dimension of token features across stages, achieving spatial downsampling while concurrently increasing the number of feature maps, as is performed in CNNs.
    Second, the linear projection prior to every self-attention block in the Transformer module is replaced with our proposed convolutional projection, which employs a s s depth-wise separable convolution operation on an 2D-reshaped token map. This allows the model to further capture local spatial context and reduce semantic ambiguity in the attention mechanism.
    
    ** What you can learn
    Then, we study the impact of each of the proposed Convolutional Token Embedding and Convolutional Projection components.
    Removing Position Embedding Given that we have introduced convolutions into the model, allowing local context to be captured, we study whether position embedding is still needed for CvT.
    In this work, we have presented a detailed study of introducing convolutions into the Vision Transformer architecture to merge the benefits of Transformers with the benefits of CNNs for image recognition tasks.

*
* (arXiv 2022.03) Transformer-based Multimodal Information Fusion for Facial Expression Analysis, 【[Paper](https://arxiv.org/pdf/2203.12367.pdf)】
* (arXiv 2022.03) Facial Expression Recognition with Swin Transformer,【[Paper](https://arxiv.org/pdf/2203.13472.pdf)】
* (arXiv 2022.01) Training Vision Transformers with Only 2040 Images,【[Paper](https://arxiv.org/pdf/2201.10728.pdf)】
* (arXiv 2021.09) Sparse Spatial Transformers for Few-Shot Learning, 【[Paper](https://arxiv.org/pdf/2109.10057.pdf)】,【[Code](https://github.com/chenhaoxing/SSFormers)】
* (arXiv 2021.06)How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers,【[Paper](https://arxiv.org/pdf/2106.10270.pdf)】 
* (arXiv 2021.03) Vision Transformers for Dense Prediction,【[Paper](https://arxiv.org/pdf/2103.13413.pdf)】,【[Code](https://github.com/isl-org/DPT)】
* (arXiv 2021.03) Face Transformer for Recognition, 【[Paper](https://arxiv.org/pdf/2103.14803.pdf)】
* (arXiv 2021.03) Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, 【[Paper](https://arxiv.org/pdf/2103.14030.pdf)】,【[Code](https://github.com/microsoft/Swin-Transformer)】
* (ICLR'21) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, 【[Paper](https://arxiv.org/pdf/2010.11929.pdf)】,【[Code](https://github.com/google-research/vision_transformer)】
* (arXiv 2017.06v1)Attention Is All You Need, 【[Paper](https://arxiv.org/pdf/1706.03762.pdf)】
### 中文文献
* 人脸活体检测，蒋方玲等【[Paper](https://kns.cnki.net/kcms/detail/detail.aspx?doi=10.16383/j.aas.c180829)】
* 人脸活体检测，马玉琨等【[Paper](https://cf.cnki.net/kcms/detail/detail.aspx?filename=KXTS202107003&dbcode=XWCJ&dbname=XWCTLKCJFDLAST2021&v=)】

  **
