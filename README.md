# video-analytics-paper

## 📚 目录（Table of Contents）

- [AccDecoder: Accelerated Decoding for Neural-enhanced Video Analytics](#accdecoder-accelerated-decoding-for-neural-enhanced-video-analytics)
- [Accelerated Neural Enhancement for Video Analytics With Video Quality Adaptation](#accelerated-neural-enhancement-for-video-analytics-with-video-quality-adaptation)
- [Spatialyze: A Geospatial Video Analytics System with Spatial-Aware Optimizations](#spatialyze-a-geospatial-video-analytics-system-with-spatial-aware-optimizations)



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






