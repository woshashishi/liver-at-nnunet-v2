# v2 Plans 感知解析规则

## 1. Attention 阶段选择

1. 从 features_per_stage 获取网络阶段总数 n_stages。
2. Stage 0 不插入 Attention。
3. 最后一个阶段 n_stages-1 为 bottleneck，不作为 skip Attention 阶段。
4. 有效 skip 阶段为 1 到 n_stages-2。
5. 选择最后两个有效 skip 阶段。
6. Dataset003_Liver 共 6 个阶段，因此解析结果为 Stage 3 和 Stage 4。

## 2. 阶段 spacing

Stage i 的物理 spacing 由输入 spacing 与截至该阶段的累计 strides 相乘得到。

stage_spacing[i] =
input_spacing × product(strides[0:i+1])

## 3. 各向异性判定

优先检查当前阶段的 nnU-Net kernel：

- 若某轴 kernel 为 1，则该轴被视为各向异性轴。
- 若所有轴 kernel 均为 3，则计算阶段 spacing 比值。
- max(stage_spacing) / min(stage_spacing) >= 2.0 时判定为各向异性。
- 否则判定为各向同性。

阈值 2.0 是本项目预先冻结的实现规则，不根据 Fold 0 Dice 调整。

## 4. CBAM 空间卷积核

- 各向同性阶段使用 3x3x3。
- 各向异性阶段在低分辨率轴使用 1，其余轴使用 3。
- 输出可能为 1x3x3、3x1x3 或 3x3x1。
- B-v2 与 D-v2 必须调用同一个解析函数。

## 5. Transformer 解析规则

- 插入阶段固定为 bottleneck，即 n_stages-1。
- blocks = 1。
- heads = 4。
- embedding_dim = min(bottleneck_channels, 256)。
- ffn_dim = 2 × embedding_dim。
- 动态三维卷积位置编码。
- residual_scale 初始化为 0。
- C-v2 与 D-v2 必须调用同一个构建函数。

## 6. Dataset003_Liver 预期结果

- Attention stages: 3, 4。
- Stage 3 spatial kernel: 3x3x3。
- Stage 4 spatial kernel: 3x3x3。
- Transformer stage: 5。
- Transformer embedding: 256。
- Transformer FFN: 512。
