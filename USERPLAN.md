# 核心结论

你列出的三种场景应当被定义为**三种不同的评测模式**，不能继续全部塞进现有的 60→120 端点参考逻辑中。

当前 `codex/norefeval-p0-reliability` 分支的实际状态是：

| 场景                      | 当前支持度 | 结论               |
| ----------------------- | ----: | ---------------- |
| 单条无参考 60 FPS 视频         |    很低 | 当前不支持，需要新增 NR 模式 |
| 单条无参考 120 FPS 视频        |    很低 | 当前不支持，需要新增 NR 模式 |
| 60 FPS 参考 + 120 FPS 插帧  |    较高 | 当前项目的核心模式，基本可用   |
| 60 FPS 原始参考 + 60 FPS 插帧 | 无直接支持 | 若逐帧对齐，改动不大，值得实现  |

**第三种场景不建议放弃。**在为第一种场景完成必要的多模式重构后，第三种场景的增量工作量会比较小，而且由于它具有完整的同帧参考，理论上反而比无参考模式更容易做准。

---

# 一、该分支本轮修改的评价

这一分支确实解决了上一轮多数 P0 可靠性问题，包括：

* 源缓存按照光流后端、权重、分辨率、遮挡参数和算法版本隔离；
* `compare` 正确传递光流后端，并隔离失败候选；
* 局部丢帧和重复帧通过单调匹配恢复；
* 对齐不可靠时 fail-closed；
* Audit 会重新构建原分辨率源端 flow、camera、edge 和人物 ROI；
* 严格重算了 12 类合成坏例的定位结果，不再保留之前被高估的数据。

这些改动都真实进入了执行路径，而不只是文档声明。源缓存契约现在包含后端和算法身份，避免 RAFT 与 Farneback 缓存混用。 对齐模块也已经加入局部单调匹配，并显式记录 drop/duplicate 事件；无法可靠恢复时会要求主流程失败关闭。

不过，这些修改改善的是：

> **已有 2× Endpoint-Referenced 模式的可靠性。**

它们没有改变项目最底层的任务定义。当前 Schema 仍明确规定：

[
Y_{2i}=X_i,\qquad Y_{2i+1}=M_i
]

即偶数帧是源视频锚点，奇数帧是生成帧。 窗口选择也只从 `alignment.generated_centers()` 中选择生成帧中心。

所以该分支并没有天然变成一个通用的“任意 FPS 无参考视频质量评测器”。

---

# 二、场景一：无参考的 60 FPS 或 120 FPS 视频

## 结论

**当前版本不支持这一场景，也不能通过简单地省略 `source` 参数来运行。**

当前 API 和 CLI 都强制要求同时提供 source 与 candidate。

当前很多核心指标也依赖已知的锚点身份：

* flow composition 需要 (X_i)、(M_i)、(X_{i+1})；
* reverse anchor cycle 需要两个生成帧反推已知锚点；
* freeze-copy 需要判断 (M_i) 是否复制 (X_i)；
* parity 明确假设偶数帧是原始帧、奇数帧是生成帧；
* anchor integrity 需要候选偶数帧和 source 对应。

例如 parity 模块直接将 120 FPS 的偶数帧定义为原始帧、奇数帧定义为生成帧。 其窗口逻辑也固定比较 anchor 子序列和 generated 子序列。

此外，当前所谓的通用 NR-VQA 后端仍然只是接口，`auto` 会直接返回 `None`，FAST-VQA、DOVER 或 VFIPQA 都尚未接入。

因此，不能认为目前已经可以“比较好地正确评估”单条 60 FPS 或 120 FPS 视频。

---

## 无参考场景能够评价什么

完全无参考时，可以比较可靠地检测：

* 闪烁和清晰度交替；
* 重复帧、冻结和突然跳帧；
* 运动补偿后残差；
* 光流轨迹不平滑；
* 局部撕裂和 flow folding；
* 人物、武器、细线的短时间抖动；
* UI 和文字的屏幕坐标不稳定；
* 编码噪声、模糊和块效应。

但无参考方法无法严格证明：

* 某个物体是否走在真实正确的轨迹上；
* 显露背景内容是否符合真实世界；
* 一个清晰且时序平滑的 hallucination 是否正确；
* 某一帧应当使用前状态还是后状态；
* 被错误删除的结构本来是否应该存在。

因此，无参考模式应输出的是：

> **时序稳定性与伪影风险分数**

而不是宣称得到了真实的插帧误差。

---

## 推荐实现：双相位自参考 + 通用时序分支

无参考 60 FPS 和 120 FPS 可以共享一套方案。

对于任意输入视频：

[
Y_0,Y_1,\ldots,Y_N
]

构造两套虚拟端点参考。

### 相位 0

```text
锚点：Y0, Y2, Y4, ...
中间：Y1, Y3, Y5, ...
```

### 相位 1

```text
锚点：Y1, Y3, Y5, ...
中间：Y2, Y4, Y6, ...
```

这样：

* 对 120 FPS 视频，相当于分别按照两个相位构造虚拟 60→120；
* 对 60 FPS 视频，相当于分别按照两个相位构造虚拟 30→60；
* 不需要预先知道哪一相位属于真实帧、哪一相位属于生成帧；
* 当前 flow composition、cycle、edge support 等大量模块可以继续复用。

两相位之外，还需要加入不依赖锚点身份的通用指标：

* 时间归一化的 motion-compensated residual；
* flow velocity、acceleration 和 jerk；
* phase-invariant alternation energy；
* 重复帧与 freeze 检测；
* track smoothness；
* UI/text temporal stability；
* 一个真正接入的 learned NR-VQA 弱先验。

### 60 FPS 与 120 FPS 不能只用相同帧数窗口

当前五帧窗口在：

* 60 FPS 下覆盖约 66.7 ms；
* 120 FPS 下只覆盖约 33.3 ms。

所以新模式必须按照**时间跨度**选择窗口，而不是固定相邻五帧。建议至少使用两个统一时间尺度：

```text
短尺度：约 16.7 ms 或 1/60 秒
中尺度：约 33.3 ms 或 1/30 秒
```

120 FPS 可以利用更多中间采样检测高频闪烁；60 FPS 则主要依赖较长时间尺度。否则 60 和 120 的分数不能放在同一量纲上。

---

## 对无参考 60 和 120 的能力判断

| 能力         | 60 FPS 无参考 | 120 FPS 无参考 |
| ---------- | ---------: | ----------: |
| 模糊、噪声、块效应  |         较强 |          较强 |
| 重复帧和冻结     |         较强 |          较强 |
| 清晰度与边缘交替   |         中等 |          较强 |
| 剑尖、UI 高频闪烁 |         中等 |          较强 |
| 运动轨迹平滑性    |         中等 |          较强 |
| 撕裂和运动层错误   |         中等 |        中等偏强 |
| 真实中间内容正确性  |          弱 |           弱 |
| 显露背景正确性    |          弱 |           弱 |

120 FPS 提供的时序采样更多，所以无参考诊断能力通常比 60 FPS 更强，但两者都不能替代真实参考。

---

# 三、场景二：60 FPS 参考 + 120 FPS 插帧视频

## 结论

这是当前框架最成熟、也最符合原始设计的模式。

这里的 60 FPS 视频不是完整 Ground Truth，而是：

> **Endpoint Reference / Reduced Reference**

它能够提供每个生成中间帧前后的两个真实锚点，因此可以使用：

* 锚点完整性；
* 双向 flow composition；
* reverse anchor cycle；
* endpoint edge support；
* freeze-copy；
* UI 静态端点约束；
* 人物和细物体的端点投影；
* 同源候选排序。

该分支新加入的单调对齐还能够恢复少量局部 drop/duplicate，并把受影响 pair 排除；超过可靠阈值时则 fail-closed。

## 当前达成程度

从工程架构上，这一模式已经基本完成。需要继续解决的是**评测器本身的真实数据标定**，而不是基础流程。

当前严格的 12 类合成坏例结果为：

* Recall：0.667；
* Precision：0.229；
* F1：0.340。

仓库已明确承认这一低 precision 是真实剩余问题，而不是生产可用结果。

因此目前适合：

* 内部模型相对排序；
* 坏例候选筛查；
* 回归结果辅助分析；
* 人工审核前的风险片段定位。

暂时不适合：

* 仅凭总分自动决定上线；
* 把错误类型标签当作精确诊断真值；
* 不经人工复核地判定人物、武器或 UI 的具体问题。

---

# 四、场景三：60 FPS 原始视频 + 60 FPS 插帧视频

## 首先要区分两种语义

### 情况 A：原始 60 FPS 是逐帧 Ground Truth

例如：

* 真实采集到 60 FPS；
* 从中抽取 30 FPS；
* 模型从 30 FPS 插回 60 FPS；
* 现在把模型输出与原始 60 FPS 进行比较。

这时两条视频具有相同时间戳，每一帧都有真实参考。

这属于：

> **Full-Reference Same-Rate Evaluation**

这种情况实现难度并不大，而且评测可靠性会高于前两个模式。

### 情况 B：两条 60 FPS 视频并不逐帧对应

例如：

* 开始时间不同；
* 有剪辑、掉帧或速度变化；
* 两条视频只是相同场景，但不是同一次录制；
* 输出经过重新定时或全部帧重生成。

这种情况需要复杂内容对齐，工作量会显著增加。

以下工作量估计以**情况 A：逐帧对应**为前提。

---

## 可以使用的指标

这一模式不应强行复用当前 `Y_{2i}/Y_{2i+1}` 结构，而应直接使用真正的 Full-Reference 指标：

### 空间参考误差

* Y/RGB L1、Charbonnier；
* PSNR；
* SSIM 或 MS-SSIM；
* LPIPS 或其他感知特征距离；
* 边缘 precision、recall 和 Chamfer；
* 人物、UI、文字等 ROI 内的局部误差。

### 时序参考误差

比较 reference 和 candidate 的帧间变化：

[
\Delta X_t=X_{t+1}-X_t
]

[
\Delta Y_t=Y_{t+1}-Y_t
]

并计算：

* temporal difference error；
* reference flow 与 candidate flow 差异；
* trajectory deviation；
* motion-compensated residual 差异；
* flicker excess；
* structure persistence error。

### 现有模块的复用

可以直接复用：

* VideoReader 与 PTS 对齐；
* scene-cut；
* cheap scan；
* 风险窗口；
* flow backend；
* UI、文字和结构 ROI；
* report、badcase 和 fusion 基础设施。

不应复用：

* anchor/generated 奇偶定义；
* endpoint flow composition；
* reverse anchor cycle；
* freeze-copy 的半程公式；
* anchor-vs-generated parity；
* `pair_of_candidate` 数据结构。

---

## 工作量评估

在完成前两个模式所需的多模式重构后，第三种场景的增量并不大。

| 工作                  |              估计 |
| ------------------- | --------------: |
| 1× PTS/图像单调对齐       |        1～2 个工程日 |
| Full-reference 空间指标 |        1～2 个工程日 |
| 时序差分、flow 与轨迹参考指标   |        2～4 个工程日 |
| 独立融合 Schema 和报告     |        1～2 个工程日 |
| 单元测试和合成验证           |        2～4 个工程日 |
| **功能版本合计**          | **约 7～12 个工程日** |

如果两条视频不是逐帧对应，需要增加：

* 局部 DTW；
* scene-level matching；
* 时间伸缩；
* crop、scale 和颜色归一化；
* 无法匹配区域的置信度管理。

这会把工作量提高到约 **3～5 周**，且准确性更难保证。

所以在通常的逐帧 GT 场景下，第三种不属于大改动，不建议删除。

---

# 五、推荐的最终模式划分

不要继续在当前 `evaluate_vfi()` 中增加更多 `if fps_ratio == ...`。建议明确拆成三个执行器。

```python
evaluate_no_reference(
    candidate_video,
)

evaluate_endpoint_reference(
    reference_video,
    candidate_video,
)

evaluate_full_reference(
    reference_video,
    candidate_video,
)
```

统一入口只负责分发：

```python
evaluate(
    candidate_video=...,
    reference_video=None,
    mode="no_reference",
)
```

推荐的 CLI 是：

```bash
# 单视频无参考，支持 60/120 FPS
rr-vfiqa evaluate \
    --mode no-reference \
    --candidate video.mp4

# 60 FPS 锚点参考 + 120 FPS 输出
rr-vfiqa evaluate \
    --mode endpoint-2x \
    --reference source_60.mp4 \
    --candidate output_120.mp4

# 60 FPS 完整参考 + 60 FPS 输出
rr-vfiqa evaluate \
    --mode full-reference \
    --reference ground_truth_60.mp4 \
    --candidate output_60.mp4
```

不建议一开始完全依赖自动推断，因为：

* 一条视频只能推断为 NR；
* 60+120 大概率是 endpoint-2x；
* 60+60 既可能是完整参考，也可能只是两个无关版本。

显式 mode 可以避免静默使用错误的数学假设。

---

# 六、三种模式的指标关系

| 指标                        | NR 60/120 | 60→120 Endpoint | 60→60 Full Reference |
| ------------------------- | --------: | --------------: | -------------------: |
| 全局模糊、噪声、压缩                |        支持 |              支持 |                   支持 |
| 运动补偿时序稳定性                 |        支持 |              支持 |                   支持 |
| 轨迹 jerk 与抖动               |        支持 |              支持 |                   支持 |
| 两相位交替检测                   |        支持 |         已知相位，更强 |                   可选 |
| Endpoint flow composition |     自参考近似 |             强约束 |                  不需要 |
| Reverse anchor cycle      |     自参考近似 |             强约束 |                  不需要 |
| Anchor integrity          |       不可用 |              支持 |             替换为直接 FR |
| 真实像素/感知误差                 |       不可用 |          只在锚点可用 |                 全帧可用 |
| 显露内容真实性                   |         弱 |              中等 |                    强 |
| 绝对质量可信度                   |        最低 |              中等 |                   最高 |

---

# 七、总体工作量与优先级

## 必做场景一：无参考 60/120

若目标是一个结构正确、能够长期扩展的版本，而不只是临时包装：

* 多模式架构与通用时间窗口：约 5～8 个工程日；
* 双相位自参考与通用时序指标：约 6～10 个工程日；
* FPS 时间归一化和独立融合：约 4～6 个工程日；
* 测试、性能和报告改造：约 4～7 个工程日。

**功能实现总计约 3～5 个工程周。**

真实数据收集和标定是额外工作，通常还需要数周。没有这部分，NR 总分只能作为启发式风险分。

## 必做场景二：60→120 Endpoint

核心已经存在。把它迁移到新的多模式架构，并保持回归一致性，约需要：

**3～5 个工程日。**

## 可选场景三：60→60 Full Reference

在前面架构完成后：

**约 7～12 个工程日。**

因此最合理的顺序是：

1. 先抽象多模式输入和时间窗口；
2. 保持现有 Endpoint 模式结果不回退；
3. 新增 NR 60/120；
4. 最后加入 Full-Reference 60→60。

---

# 最终判断

**当前分支已经较好地修复了场景二的工程可靠性，但尚未支持场景一和场景三。**

其中：

* 场景一是一次中等规模的技术路线扩展，必须建立独立 NR 评测逻辑，不能只删除 `source` 参数；
* 场景二可以继续作为当前框架的高置信核心；
* 场景三若 reference 与 candidate 逐帧对应，工作量并不大，而且评测价值很高，建议保留。

最重要的架构原则是：**三个模式必须使用不同的特征可用性、融合权重和校准模型，不能共享同一个未经区分的 0～100 总分。**
