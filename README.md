# nnU-Net v2 肝脏及肝肿瘤分割科研项目

## 研究目标

基于 MIC-DKFZ nnU-Net v2，研究计划感知的局部—全局特征增强方法，
用于三维腹部 CT 肝脏及肝肿瘤分割。

## 数据集

- 主数据集：MSD Dataset003_Liver
- 条件性验证数据集：MSD Dataset007_Pancreas
- Pancreas 阶段仅比较标准 Baseline 与最终 Hybrid
- 当前阶段暂不启动 Pancreas

## 模型矩阵

- A：标准 nnUNetTrainer
- B：nnU-Net v2 + 各向异性感知 CBAM-lite
- C：nnU-Net v2 + 残差式 Bottleneck Transformer
- D：nnU-Net v2 + CBAM-lite + Transformer
- Extra-Conv：参数量匹配对照
- ResEnc：官方强参考模型

## 当前阶段

Phase 0–1：项目台账、环境锁定和官方 Baseline 准备。

当前任务：

1. 固定项目目录和 Git 版本；
2. 锁定 nnU-Net、Python、PyTorch 与 CUDA 环境；
3. 准备 Dataset003_Liver；
4. 完成官方 Baseline Fold 0 的训练、验证和推理闭环。

## 实验原则

- 官方 nnU-Net 核心源码保持不变；
- 自定义模块放在独立扩展包 nnunet_at_v2 中；
- A/B/C/D 使用相同的数据划分、Plans 和训练策略；
- Fold 0 只用于代码、显存和训练稳定性检查；
- 不根据 Fold 0 最终 Dice 反复选择网络结构；
- 正式五折开始后不得继续修改网络结构；
- 所有论文结果必须能够追溯到病例级 CSV；
- 不将无标签 imagesTs 用作本地定量测试集。

## 硬件策略

GPU 通过云服务器租用。

正式训练 GPU 根据以下指标选择：

- 每 iteration 时间；
- GPU 每小时租金；
- 预计单 Fold 时间；
- 峰值显存；
- 预计单 Fold 总费用。

论文中的显存、推理速度和吞吐量必须在同一型号 GPU 上统一测量。
