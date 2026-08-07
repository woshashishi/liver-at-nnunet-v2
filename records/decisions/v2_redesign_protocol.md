# v2 指南一致性重设计协议

## 当前状态

A/B/C/D v1 Fold 0 开发筛选已经完成并归档。

v1 结果显示：
- 肝脏 Dice 基本稳定；
- B、C、D 的下降主要来自肿瘤 Dice；
- CBAM residual_scale 明显偏离零；
- Transformer residual_scale 接近零；
- Hybrid 未显示明确互补收益。

## 重设计依据

v2 的修改不是根据多个 Fold 0 结果搜索最高 Dice，而是修正 v1
与科研指南预定义规格之间的偏差。

## v2 规则

1. CBAM 阶段和空间卷积核由 nnUNetPlans 动态推导。
2. B 与 D 完整共享同一 Attention 实现和配置。
3. C 与 D 完整共享同一 Transformer 实现和配置。
4. Transformer 保持在 bottleneck。
5. Transformer 使用 1 block、4 heads、embedding 256、FFN 512。
6. 不把 v1 checkpoint 中学习到的 residual_scale 用作 v2 初始化值。
7. v2 先完成 CPU、GPU、AMP、短训练、reload 和 predict 闭环。
8. 工程闭环通过后只执行一次 v2 Fold 0 开发检查。
9. 最终配置创建冻结 tag 后，才开始 A/B/C/D Fold 0-4。
10. v1 与 v2 结果不得拼接为同一五折 OOF。
