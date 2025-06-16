# video-analytics-paper

## AccDecoder: Accelerated Decoding for Neural-enhanced Video Analytics

** 本文由哥廷根大学完成，发表在 INFOCOM 2023。代码见：https://github.com/mi150/AccDecoder **

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
** 本文由哥廷根大学和南京大学共同完成，发表在 TON 2024 ,是 AccDecoder（INFOCOM 2023） 的扩展。 **

### 1.动机： 处理来自不同摄像头或自适应编码器（如 AWStream, Pensieve, Chameleon）的异构分辨率视频流。

- **核心改进**： 在 DRL 调度器的状态 (State) 中显式加入视频块的分辨率信息。

### 2.效果： 调度器能学习针对不同分辨率块的最优阈值设置。

- **对低分辨率块**：倾向于选择更多帧做 SR（成本相对低，精度提升潜力大）。

- **对高分辨率块**：更谨慎选择 SR 帧（成本高）。

### 3.优势： 显著提升了 AccDecoder 的通用性 (Generality) 和可扩展性 (Scalability)，能更好适应实际中多源、自适应码率的视频流。





