# video-analytics-paper

## 📚 目录

- [AccDecoder: Accelerated Decoding for Neural-enhanced Video Analytics](#accdecoder-accelerated-decoding-for-neural-enhanced-video-analytics)
- [Accelerated Neural Enhancement for Video Analytics With Video Quality Adaptation](#accelerated-neural-enhancement-for-video-analytics-with-video-quality-adaptation)
- [Spatialyze: A Geospatial Video Analytics System with Spatial-Aware Optimizations](#spatialyze-a-geospatial-video-analytics-system-with-spatial-aware-optimizations)
- [Boosting Neural Representations for Videos with a Conditional Decoder](#boosting-neural-representations-for-videos-with-a-conditional-decoder)
- [Region-based Content Enhancement for Efficient Video Analytics at the Edge](#region-based-content-enhancement-for-efficient-video-analytics-at-the-edge)

## AccDecoder: Accelerated Decoding for Neural-enhanced Video Analytics

**本文由哥廷根大学完成，发表在 INFOCOM 2023。代码见：https://github.com/mi150/AccDecoder**

### 1. 需要解决的问题

低质量的视频导致视频分析准确率下降，需要对视频内容进行增强从而提高准确率。常规的方法要么花费时间过多，要么无法很好地适应视频内容的动态性，导致准确率下降。因此本文需要解决的问题是：

> 如何在保证视频分析准确率的同时，尽量降低时延，达到一个良好的权衡？

### 2. 解决方案——AccDecoder

AccDecoder 将视频分析流程分为三个步骤：

- **超分辨率增强（SR）**：为了降低延迟同时保证分析准确率，本文选择少量视频帧（称为锚帧）进行超分辨率增强。其他低分辨率帧结合锚帧、运动向量和残差合成高分辨率帧。

- **DNN 推理**：为了进一步降低延迟，仅对推理帧运行 DNN，对非推理帧则重用推理结果，并结合运动向量进行调整。

- **重用结果**：结合视频连续性和先前结果，提高效率并减少冗余计算。

### 3. 如何选择用于 SR 和推理的帧？——使用马尔可夫决策过程（MDP）

- **状态（State）**：智能体决策的依据。包括关键帧的内容特征、视频块内部的帧间差异。

- **动作（Action）**：智能体根据当前状态做出的决策。每个视频块的帧根据两个阈值分类：

  - `tr1`（SR Threshold）：用于选择锚点帧（做超分增强）  
  - `tr2`（Inference Threshold）：用于选择推理帧（执行 DNN）

- **奖励（Reward）**：智能体根据在给定时延约束下的分析准确率，来衡量策略好坏，目标是最大化准确率。


## Accelerated Neural Enhancement for Video Analytics With Video Quality Adaptation
**本文由哥廷根大学和南京大学共同完成，发表在 TON 2024 ,是 AccDecoder（INFOCOM 2023） 的扩展。**

### 1.动机： 处理来自不同摄像头或自适应编码器（如 AWStream, Pensieve, Chameleon）的异构分辨率视频流。

- **核心改进**： 在 DRL 调度器的状态 (State) 中显式加入视频块的分辨率信息。

### 2.效果： 调度器能学习针对不同分辨率块的最优阈值设置。

- **对低分辨率块**：倾向于选择更多帧做 SR（成本相对低，精度提升潜力大）。

- **对高分辨率块**：更谨慎选择 SR 帧（成本高）。

### 3.优势： 显著提升了 AccDecoder 的通用性 (Generality) 和可扩展性 (Scalability)，能更好适应实际中多源、自适应码率的视频流。


## Spatialyze: A Geospatial Video Analytics System with Spatial-Aware Optimizations  
**本文由 UC Berkeley 完成，发表在 VLDB 2024，项目主页见 https://spatialyze.github.io/**

### 1. 动机：传统视频分析系统未充分利用摄像头元数据（如地理位置、时间、姿态等）

- **问题核心**：视频实际采集自真实世界的物理空间，但现有方法大多只处理图像帧本身，忽略了空间语义与物理约束。

### 2. 方法：Spatialyze 系统通过空间元数据优化视频查询流程，共包含三个阶段：

- **构建阶段**：用户定义虚拟世界（摄像头布局、地理信息等）  
- **过滤阶段**：用户以声明式语言描述感兴趣的场景（如“在十字路口出现的人”）  
- **观察阶段**：系统执行完整分析流程，包括数据整合、视频处理、对象查询和结果输出

其中，**视频处理器模块**采用了四项空间感知优化技术：

- **Road Visibility Pruner**：跳过未覆盖目标区域的相机帧  
- **Object Type Pruner**：仅处理用户关心的目标类型  
- **Geometry-Based 3D Location Estimator**：根据边界框推算目标的 3D 物理位置  
- **Exit Frame Sampler**：基于事件预测减少不必要的帧处理

### 3. 效果：相比未优化系统，Spatialyze 在保持 97.1% 准确率的同时，提升了 5.3 倍的执行效率


## Boosting Neural Representations for Videos with a Conditional Decoder
**本文由香港科技大学、商汤科技、香港中文大学等机构合作完成，发表在 CVPR 2023。项目主页见 https://github.com/Xinjie-Q/Boosting-NeRV**

### 1. 动机：现有隐式视频表示（INRs）未能充分发挥其潜力

- **问题核心**：在解码目标视频帧时，网络内部的**中间特征与目标帧对齐不充分**，且压缩流程存在**不一致性**，导致重建质量和压缩效率受限。

### 2. 方法：提出一个通用的增强框架，通过引入条件解码器和一致性压缩来提升性能

该框架主要包含四项核心技术创新：

- **时间感知的条件解码器 (Temporal-aware Conditional Decoder)**：利用**帧索引**作为条件，通过一个无归一化的**仿射变换模块 (TAT)** 动态调整中间特征，实现特征与目标帧的精确对齐。

- **正弦NeRV类块 (Sinusoidal NeRV-like Block)**：采用 **SINE (正弦)** 作为激活函数，以生成更多样化的特征，并以更少的参数提升模型容量。

- **高频信息保留的损失函数 (High-frequency-preserving Loss)**：组合**频域L1损失**、空域L1损失和MS-SSIM损失，以更好地保留视频的边缘和纹理细节。

- **一致的熵最小化压缩 (Consistent Entropy Minimization, CEM)**：采用简单的**高斯模型**替代复杂的代理网络来估计熵，保证了训练和推理阶段的**完全一致性**，从而优化了率失真性能。

### 3. 效果：在多种视频任务上显著优于基线，并达到SOTA压缩性能

- **视频回归**：在UVG数据集上，相比基线模型平均提升 **1.2-1.7 dB PSNR**，且收敛速度更快。
- **视频压缩**：RD性能显著优于传统编解码器（H.264/H.265）和部分SOTA学习方法（DCVC），展示了极强的竞争力。
- **视频修复与插值**：在修复任务上平均带来 **1.9-4.2 dB** 的性能提升，并在插值任务上表现更优。


## Region-based Content Enhancement for Efficient Video Analytics at the Edge 
**本文由清华大学、南京大学、哥廷根大学等机构合作完成，发表在  NSDI '25。项目主页见 https://github.com/mi150/RegenHance**

### 1. 动机：现有内容增强方法在边缘视频分析中存在性能瓶颈

- **问题核心**：现有内容增强方法（如超分辨率）在应用于边缘视频分析时存在**性能与精度的根本矛盾**。对**整个视频帧进行增强**会消耗海量计算资源，导致吞吐量极低，无法实时处理；而**选择性地增强部分关键帧**又会因信息复用造成精度大幅下降，无法满足AI分析任务的苛刻要求。

### 2. 方法：提出一个名为 RegenHance 的系统，通过只增强关键区域来打破性能与精度的困境

该系统通过三大创新组件，实现了高效率与高精度的平衡：

- **基于宏块的区域重要性预测 (Macroblock-based Region Importance Prediction)**：采用视频编码中的**宏块 (`Macroblock`)** 作为基本单元，通过一个超轻量级预测模型，在原始低质量视频上**快速且准确地定位**出对分析精度提升最大的关键区域。

- **区域感知的打包增强 (Region-aware Enhancement)**：将识别出的零散、不规则的关键区域高效地**“拼接打包”**成一个或多个紧凑的矩形张量。该过程被建模为一个**二维装箱问题**，只对打包后的小尺寸张量进行增强，极大降低了计算开销。

- **基于剖析的执行规划 (Profile-based Execution Planning)**：通过**离线剖析**各组件（解码、预测、增强、分析）在不同负载下的性能，在线为整个分析流水线**动态生成最优执行计划**，确保CPU/GPU资源被充分利用，最大化端到端吞吐量。

### 3. 效果：在精度和吞吐量上全面超越SOTA方法，实现显著性能提升

- **精度与吞吐量双重提升**：在目标检测和语义分割任务上，相比SOTA的帧级增强方法（如Nemo、NeuroScaler），**分析精度提升10-19%**，同时**端到端吞吐量提升2-3倍**。
- **资源利用效率极高**：相比全帧增强，可节省**高达77%的GPU计算资源**，将宝贵的算力集中用于对分析任务真正有益的区域。
- **强大的平台通用性**：在五种异构硬件平台（从云端A100到边缘Jetson）上均表现出**强大的鲁棒性和有效性**，证明了其在真实边缘计算场景中的实用价值。

