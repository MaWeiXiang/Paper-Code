# Paper&Code
Record my research and encourage myself

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
