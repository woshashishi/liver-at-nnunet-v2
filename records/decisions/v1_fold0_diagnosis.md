# A/B/C/D v1 Fold 0 开发诊断

## 实验定位

本轮 A/B/C/D Fold 0 属于开发筛选实验，用于检查：

- 网络构建与训练稳定性；
- checkpoint reload；
- sliding-window inference；
- 初步验证趋势；
- 残差门控学习状态。

在最终网络配置冻结前，不启动 B/C/D 的 Fold 1-4。

## Fold 0 指标

| 模型 | Mean Dice | Liver Dice | Tumor Dice |
|---|---:|---:|---:|
| A Baseline | 0.820035 | 0.965392 | 0.674678 |
| B CBAM v1 | 0.803780 | 0.964704 | 0.642856 |
| C Transformer v1 | 0.811648 | 0.965369 | 0.657927 |
| D Hybrid v1 | 0.804760 | 0.965150 | 0.644371 |

## 最终 residual_scale

### B CBAM v1

- Encoder Stage 3 CBAM: 2.109679
- Encoder Stage 4 CBAM: -0.633610

### C Transformer v1

- Encoder Stage 5 Transformer: -0.006148

### D Hybrid v1

- Encoder Stage 3 CBAM: 1.215555
- Encoder Stage 4 CBAM: -0.955964
- Encoder Stage 5 Transformer: 0.020387

## 初步判断

1. B、C、D 的肝脏 Dice 与 Baseline 基本接近。
2. Mean Dice 的下降主要来自肿瘤 Dice。
3. B 和 D 中 CBAM 门控明显偏离零。
4. C 和 D 中 Transformer 门控绝对值较小。
5. D 的表现接近 B，当前 Hybrid 未体现明确互补收益。
6. 这些结果用于开发诊断，不作为完整五折 OOF 最终结论。
7. v1 的代码、checkpoint、日志、分支和标签全部保留，不删除、不覆盖。
