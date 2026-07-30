# 核心结论

你的目标已经非常明确：这个项目不应该继续被当作“等待训练和标定的指标研究仓库”，而应该收敛成一个：

> **输入视频后自动选择评测流程、自动扫描、自动定位问题、输出单一主分数，并生成可直接查看的坏例视频、热力图和诊断报告的工程工具。**

这条路线不要求再训练一个评分模型，也不要求你为每批视频人工设置阈值。代价是必须把最终分数定义为：

> **确定性的工程质量分数与风险诊断分数，而不是声称等价于人类主观 MOS。**

当前三模式架构已经适合作为这个产品的基础，而且三种模式本来就具有独立的输入条件和分数语义，不能跨模式直接比较。

接下来最重要的事情不再是增加大量新指标，而是完成四个产品闭环：

1. **一个命令完成自动评测；**
2. **一个稳定的主分数和明确的子分数；**
3. **把分数变成可解释的问题诊断；**
4. **把问题通过热力图、轨迹图和坏例视频直接展示出来。**

---

# 一、最终使用体验应该是什么

建议最终只保留一个主要入口：

```bash
rr-vfiqa inspect \
  --candidate output.mp4 \
  --reference reference.mp4 \
  --out result
```

无参考时：

```bash
rr-vfiqa inspect \
  --candidate output.mp4 \
  --out result
```

用户不需要选择：

* 窗口数量；
* 光流宽度；
* 阈值；
* 具体劣化类型；
* 是否运行哪一个指标；
* 是否需要 Audit；
* 哪些区域需要检查。

系统自动完成以下过程：

```text
输入检查
    ↓
安全模式判断
    ↓
低成本全片扫描
    ↓
风险窗口选择
    ↓
精细指标计算
    ↓
高风险窗口原分辨率复核
    ↓
证据融合与问题分类
    ↓
HTML 报告、热力图和坏例视频
```

---

# 二、模式可以自动判断，但必须是“安全自动判断”

你不希望每次手工选择模式是合理的，但项目不能无条件猜测模式。

建议新增：

```text
mode = auto-safe
```

## 自动判断规则

### 只有 candidate

直接选择：

```text
no-reference
```

### reference 与 candidate 的 FPS 约为 1:2

例如：

```text
reference 60 FPS
candidate 120 FPS
```

自动选择：

```text
endpoint-2x
```

### reference 与 candidate FPS 相同

先运行低成本的 same-rate alignment。

只有满足以下条件才自动进入：

```text
full-reference
```

* FPS 比约等于 1；
* 时长相近；
* PTS 可以单调匹配；
* 内容描述符误差足够低；
* 匹配覆盖率足够高；
* 几何策略合法。

当前 FR alignment 已经使用图像误差、匹配覆盖率和局部 gap 数量进行 fail-closed，具备实现安全自动判断的基础。

如果两条 60 FPS 视频不是逐帧对应的同一次录制，程序应明确返回：

```text
无法安全确定为 Full Reference。
两条视频时间相近，但逐帧内容不对应。
未生成误导性分数。
```

这比静默选择错误模式更重要。

---

# 三、不要再让用户选择 Fast、Standard、Audit

当前项目存在三个 preset，但最终用户不应该理解这些内部资源策略。

建议对外只保留：

```text
--speed fast
--speed balanced
--speed thorough
```

默认不传时使用：

```text
balanced
```

更好的方式是直接使用自动三级级联。

## Tier 1：全片低成本扫描

所有帧或稀疏帧计算：

* 清晰度；
* 帧差；
* duplicate/freeze；
* 场景切换；
* 编码质量；
* 简化边缘；
* 简化参考误差；
* 轻量内容指纹。

## Tier 2：高风险和均匀窗口精评

自动选择：

* 一部分全片均匀窗口；
* 一部分高风险窗口；
* 场景切换附近窗口；
* 运动高峰窗口；
* UI 高密度窗口。

## Tier 3：原分辨率自动复核

不要求用户指定 Audit。

当窗口满足以下任意条件时自动升级：

* 风险分数进入全片前 5%；
* 多个指标同时触发；
* 指标之间发生矛盾；
* confidence 较低；
* 小目标或 UI 区域占比过小；
* 低分辨率结果接近阈值。

这样，正常视频的运行成本接近 Standard，而真正可疑的位置自动获得 Audit 质量。

---

# 四、主分数设计

用户最终应该看到一个主分数，但内部必须保留子分数和置信度。

报告顶部建议固定显示：

```text
Overall Quality       72.8 / 100
Confidence            0.83
Mode                  Full Reference
Quality Level         Noticeable Issues
Affected Duration     7.4%
Worst Interval        00:12.34–00:12.62
```

## 分数必须同时包含三层信息

### 第一层：Overall Score

提供快速判断。

```text
90–100  很好
80–90   基本良好
65–80   有可见问题
45–65   明显问题
0–45    严重问题
```

这些只能称为工程风险等级，不能称为人类 MOS。

### 第二层：Subscores

例如 NR：

```text
Temporal Stability
Motion Smoothness
Phase Consistency
UI / Text Stability
Technical Quality
Cadence Integrity
```

Endpoint：

```text
Motion Consistency
Temporal Stability
Structural Integrity
Character Integrity
Thin Object / Weapon
UI / Text
Transition Quality
Technical Quality
```

FR：

```text
Spatial Fidelity
Structural Fidelity
Temporal Fidelity
Motion Fidelity
Color Fidelity
```

### 第三层：Confidence

置信度不能被隐藏。

以下情况必须降低 confidence：

* 覆盖窗口太少；
* 对齐接近失败阈值；
* 大量特征为 NaN；
* 指标之间互相矛盾；
* 场景切换过多；
* 全视频几乎没有运动；
* NR 无法判断内容真实性；
* 后端降级为 Farneback。

当前报告已经记录 MetricResult 的失败、warnings、coverage 和 confidence，可以继续沿用。

---

# 五、No-Reference 需要增加 Cadence Integrity

这是当前最需要补全的评分问题。

当前公共 NR 总分只融合 1/60 和 1/30 秒共同时间尺度，native cadence 只作为诊断项。这保证了 60 FPS 和 120 FPS 的公共分数不会混用不同物理跨度。

但也导致一种严重缺陷：

```text
120 FPS 视频：
Y0 = X0
Y1 = X0
Y2 = X1
Y3 = X1
...
```

它的实际有效帧率只有 60 FPS，但共同 1/60 秒尺度可能仍然正常。

因此建议增加：

```text
cadence_integrity
```

## Cadence Integrity 的证据

* native duplicate fraction；
* native freeze fraction；
* native MCT residual；
* native flow magnitude distribution；
* 相邻帧运动增量是否交替为零；
* native sharpness alternation；
* PTS cadence irregularity；
* 周期性重复模式。

## 主分数融合方式

推荐使用保守的乘法惩罚：

[
S_{\text{overall}}
==================

S_{\text{common}}
\cdot
\exp(-\lambda R_{\text{cadence}})
]

而不是把 native 特征和 common 特征简单平均。

这意味着：

* 正常 cadence 不影响分数；
* duplicate cadence 会明显降低总分；
* common-time 质量仍可单独显示；
* 60/120 公共尺度仍然保留。

报告可以同时输出：

```text
Common-time Quality   92.4
Cadence Integrity     38.7
Overall Quality       61.3
```

这样既满足直接得到一个总分，也不会掩盖 120 FPS 复制帧。

---

# 六、自动参数适应应该如何实现

你不希望针对不同视频调参数，核心不是删除参数，而是把参数变成自动归一化。

## 1. 时间尺度按 PTS 归一化

项目目前已经基本完成：

* native；
* 1/60 秒；
* 1/30 秒；
* 固定物理时间窗口。

这部分方向正确。

## 2. 空间尺度按图像尺寸归一化

以下特征不能直接使用像素：

* Chamfer 距离；
* 轨迹误差；
* 边缘偏移；
* UI 漂移；
  -物体长度；
  -局部位移。

应统一转为：

[
d_{\text{normalized}}
=====================

\frac{d_{\text{pixels}}}
{\sqrt{H^2+W^2}}
]

或者除以工作分辨率宽度。

否则 480p、1080p 和 4K 会使用不同量纲。

## 3. 运动误差按真实运动强度归一化

例如：

[
E_{\text{flow}}
===============

\frac{\lVert F_{\text{candidate}}-F_{\text{reference}}\rVert}
{\lVert F_{\text{reference}}\rVert+\epsilon}
]

当前 FR 已经在 flow error 中使用了类似归一化。

## 4. 同时使用绝对阈值和视频内自适应阈值

只使用视频内 percentile 会漏掉“全片一直很差”的情况。

只使用绝对阈值又会对不同游戏和画面风格过敏。

建议风险值取两者最大值：

[
R(x)
====

\max
\left(
R_{\text{absolute}}(x),
R_{\text{relative}}(x)
\right)
]

其中相对风险使用 robust statistics：

[
z =
\frac{x-\operatorname{median}(x)}
{1.4826\cdot\operatorname{MAD}(x)+\epsilon}
]

这样既可以发现：

* 全片稳定模糊；
* 局部突然变坏；
* 某几个窗口的异常；
* 不同游戏整体纹理差异。

---

# 七、项目下一步最大的产品功能：诊断证据引擎

用户真正需要的不是只看到：

```text
overall = 63.4
```

而是看到：

```text
00:12.34–00:12.62
严重程度：高
可能问题：生成帧复制前一帧
证据：
- 相邻帧差接近 0
- native flow 接近 0
- 1/60 秒运动仍然存在
- 奇偶帧运动增量明显不对称
```

建议新增统一诊断结构：

```python
@dataclass
class DiagnosticIssue:
    issue_type: str
    title: str
    severity: float
    confidence: float
    start_time: float
    end_time: float
    evidence: list[Evidence]
    probable_causes: list[str]
    maps: list[str]
    clips: list[str]
```

## 诊断必须基于多个证据组合

不要根据单个特征直接下结论。

### Duplicate / Freeze

组合证据：

* native frame L1 极低；
* native flow 极低；
* 前后较长时间尺度有运动；
* 重复模式持续；
* 非静态场景。

可能原因：

* 模型复制端点；
* 视频编码或导出丢帧；
* 推理循环重复写帧；
  -时间戳错误。

### Generated-frame Blur

组合证据：

* 奇偶清晰度差；
* 奇偶边缘密度差；
* common MCT 升高；
* Endpoint/FR 中生成帧空间误差升高。

可能原因：

* 插帧结果过度平滑；
* blend mask 过软；
* motion compensation 不准确；
  -编码质量不足。

### Ghosting / Double Exposure

组合证据：

* edge precision 下降；
  -额外双边缘；
* composition error；
* cycle residual；
* MCR residual；
  -局部高频能量增加。

可能原因：

* 双向 warp 错位；
* blend mask 错误；
  -遮挡处理失败；
  -前后运动层混合。

### Tearing / Flow Folding

组合证据：

* Jacobian determinant 过低；
* folding ratio；
* divergence/curl 异常；
  -局部 edge discontinuity；
* composition error 集中。

可能原因：

-光流局部错误；
-运动边界归属错误；
-形变过强；
-遮挡区域错误传播。

### UI / Text Instability

组合证据：

* screen-static ROI；
* edge XOR；
  -组件数量交替；
  -笔画断裂；
  -位置来回漂移；
  -背景区域相对稳定。

可能原因：

* UI 被当成场景运动；
* UI 未单独处理；
  -文字区域融合或缩放错误；
  -奇偶帧处理不一致。

### FR Color Pipeline Mismatch

组合证据：

* chroma L1 高；
* Y L1 较低；
  -结构和 flow 基本正常。

可能原因：

-色彩空间转换；
-RGB/BGR 错误；
-limited/full range；
-编码器色彩矩阵；
-gamma 差异。

“可能原因”必须明确标记为推断，不应声称知道模型内部真实根因。

---

# 八、把 MetricResult 的 maps 和 instances 真正使用起来

当前 `MetricResult` 已经定义：

* `scalars`；
* `maps`；
* `instances`；
* `coverage`；
* `confidence`；
* `warnings`。

但执行器目前主要使用 scalars。

下一步所有核心指标都应该返回 error map。

## NR 应输出的图

* `mct_residual_map`；
* `composition_error_map`；
* `self_cycle_residual_map`；
* `flow_fold_map`；
* `jacobian_determinant_map`；
* `phase_sharpness_map`；
* `ui_edge_instability_map`；
* `track_trajectory_overlay`；
* `duplicate_frame_indicator`。

## Endpoint 应输出的图

* `endpoint_composition_map`；
* `reverse_cycle_map`；
* `edge_support_loss_map`；
* `ghost_edge_map`；
* `anchor_integrity_map`；
* `character_proxy_map`；
* `thin_object_map`；
* `ui_static_mask`；
* `text_stroke_map`。

## FR 应输出的图

* `absolute_rgb_error_map`；
* `luma_error_map`；
* `ssim_error_map`；
* `gradient_error_map`；
* `edge_false_negative_map`；
* `edge_false_positive_map`；
* `flow_difference_map`；
* `camera_residual_flow_map`；
* `temporal_difference_map`；
* `flicker_map`。

这些 map 不应该全部保存整片，只保存：

* top risk 窗口；
  -每种 issue 最严重的若干窗口；
  -用户指定时间段。

---

# 九、可视化输出应该升级为 HTML 诊断报告

当前输出主要是：

* `report.json`；
* `timeline.md`；
* `timeline.png`；
* `badcases/*.mp4`。

这对开发者有用，但不够直观。

建议新增：

```text
report.html
```

## HTML 顶部

```text
总分
置信度
模式
视频信息
主要问题数量
受影响时间比例
最严重问题
```

## 时间线

一条可交互时间线：

```text
红色：严重
橙色：中等
黄色：轻微
灰色：场景切换或不确定区域
```

不同 issue 使用不同轨道：

```text
Temporal
Motion
Structure
UI/Text
Technical
Cadence
```

点击时间线直接跳转坏例视频。

## 问题卡片

每个问题包含：

```text
问题名称
时间段
严重程度
置信度
证据
可能原因
原始视频
热力图叠加视频
关键帧
```

## Reference 模式展示

Endpoint：

```text
Previous endpoint | Candidate middle | Next endpoint
```

FR：

```text
Reference | Candidate | Error map | Overlay
```

NR：

```text
Candidate | Motion-compensated prediction | Residual | Flow/track overlay
```

## 报告目录

建议最终输出：

```text
result/
├── report.html
├── report.json
├── summary.json
├── timeline.csv
├── timeline.png
├── badcases/
│   ├── issue_001_original.mp4
│   ├── issue_001_overlay.mp4
│   └── issue_001_compare.mp4
├── heatmaps/
│   ├── issue_001_mct.png
│   ├── issue_001_flow.png
│   └── issue_001_edge.png
├── thumbnails/
└── debug/
```

---

# 十、坏例视频应该如何生成

坏例视频不能只是截取原视频。

每个坏例建议生成三类文件。

## 1. Original Clip

仅截取问题时间段，前后增加约 0.5 秒上下文。

## 2. Overlay Clip

在原视频上叠加：

* 热力图；
  -轨迹；
  -异常区域框；
  -问题名称；
  -分数；
  -时间戳。

## 3. Diagnostic Compare Clip

根据模式不同生成。

### NR

```text
原视频 | 运动补偿预测 | 残差热力图
```

### Endpoint

```text
前端点 | 中间候选 | 后端点 | composition/cycle map
```

### FR

```text
Reference | Candidate | Error heatmap
```

这比只输出静态 PNG 更容易判断问题是否真实可见。

---

# 十一、需要一个批量评测入口

你描述的真实工作流不是只评测一个视频，而是“来了一些视频”。

建议新增：

```bash
rr-vfiqa inspect-batch \
  --manifest jobs.json \
  --out runs/2026-07-30
```

Manifest 只保存输入关系：

```json
{
  "items": [
    {
      "id": "model_a_case_001",
      "candidate": "a.mp4",
      "reference": "ref.mp4"
    },
    {
      "id": "single_video_002",
      "candidate": "b.mp4"
    }
  ]
}
```

模式由 auto-safe 判断。

输出一个批量首页：

```text
ID
Mode
Overall
Confidence
Worst issue
Affected duration
Open report
```

多个模型只有在满足：

* 相同模式；
  -相同 reference；
  -相同 FPS；
  -同源内容；

时才自动排名。

---

# 十二、不做人为评分时，如何验证项目不是“自己骗自己”

你不希望做人工打分和训练，这是可以接受的，但仍然需要验证指标的基本正确性。

最适合你的方法不是人工 MOS，而是：

> **Metamorphic Testing，变形测试。**

利用你已有的真实视频，程序自动注入已知劣化。

## 自动注入类型

* 某个时间段 freeze；
* 单帧 duplicate；
* 交替帧 blur；
  -局部高斯模糊；
* ghost 双重曝光；
  -局部 spatial shift；
* tearing；
* UI 区域抖动；
  -文字断裂；
  -色彩偏移；
  -噪声；
  -块效应；
  -帧率或 PTS 异常。

然后自动检查：

1. 劣化后总分是否下降；
2. 对应子分是否下降；
3. 最差窗口是否覆盖注入时间段；
4. error map 是否覆盖注入区域；
5. 问题类型是否合理；
6. 劣化越强，分数是否单调下降。

这不需要人工打标签，也不需要训练模型。

真实视频只作为干净基底，缺陷位置和类型由程序自动知道。

这是后续最重要的测试体系。

---

# 十三、剩余路线重新排序

## P0：完成开箱即用体验

### 目标

一个命令输入视频，直接得到完整报告。

### 实现内容

1. `auto-safe` 模式路由；
2. 自动 Tier 1/2/3 预算；
3. `cadence_integrity`；
   4.统一问题对象 `DiagnosticIssue`；
4. HTML 报告；
   6.坏例原视频、Overlay 和 Compare Clip；
   7.批量评测入口；
   8.所有异常 fail-closed，并输出易懂原因。

这是下一阶段的最高优先级。

---

## P1：完成 error map 与诊断规则

### 实现内容

1. 让 NR/Endpoint/FR metric 返回 `MetricResult.maps`；
2. 建立统一 map normalization；
   3.建立问题证据规则；
   4.生成区域 box、track 和 mask；
   5.把证据与可能原因写入报告；
   6.建立 issue-level confidence；
   7.同一时间段多证据合并。

完成后，项目才真正从“打分器”变成“诊断器”。

---

## P2：修正剩余算法问题

1. NR cadence failure 进入最终判定；
2. NR flow geometry 使用 affine/homography camera residual；
   3.动态 UI 单调变化不直接扣分；
3. FR gap 处重置 temporal previous；
4. FR UI ROI 扩展到四边和中心；
5. Endpoint semantic category 增加 feature、instance 和 coverage gate；
   7.所有空间距离统一归一化。

---

## P3：性能和缓存

1. FR reference descriptor cache；
2. FR reference flow cache；
3. FR reference ROI cache；
4. NR 跨窗口 flow pair cache；
   5.顺序解码复用；
   6.批量 RAFT；
   7.分阶段时间和 RAM/VRAM 报告；
   8.长视频断点续跑；
   9.失败窗口可重试；
   10.结果缓存按视频 hash 和算法 contract 隔离。

---

## P4：无需人工打分的真实视频回归体系

建立：

```text
tests/real_corpus/
```

不需要人工评分，只需要保存：

-真实视频路径或 manifest；
-自动注入缺陷配置；
-预期时间段；
-预期区域；
-预期子分方向；
-最低分数下降幅度。

每次提交自动跑：

```text
clean > degraded
轻度 > 中度 > 重度
定位窗口覆盖注入区域
热力图覆盖注入区域
```

---

# 十四、不建议继续投入的方向

## 1. 暂时不要训练总分模型

既然你的目标是不维护训练数据，那么现在继续开发 LightGBM 或新的评分网络不是主路径。

Endpoint 的校准器可以保留为可选功能，但默认仍使用确定性公式。

## 2. 不要接入大量通用 NR-VQA

FAST-VQA、DOVER、NIQE 等可以作为弱先验，但不能成为游戏插帧诊断的核心。

原因是它们往往更关注：

-自然视频观感；
-编码；
-构图；
-整体质量；

而不一定准确识别：

-奇偶帧模糊；

* UI 抖动；
  -剑尖闪烁；
  -运动层错误；
  -生成帧复制。

当前默认关闭 VQA 后端是正确的。

## 3. 不要继续无限增加标量特征

新增一个特征只有在满足以下至少一项时才有价值：

-可以定位新的问题类型；
-可以生成有意义的 error map；
-可以显著降低误报；
-可以改善某种已知缺陷方向性；
-可以提高置信度判断。

否则只会让总分更难解释。

---

# 十五、建议的最终模块结构

```text
rr_vfiqa/
├── inspect.py
├── mode_router.py
├── execution/
│   ├── auto_budget.py
│   ├── cascade.py
│   └── batch_runner.py
├── diagnosis/
│   ├── schema.py
│   ├── evidence.py
│   ├── rules.py
│   ├── issue_merger.py
│   └── cause_inference.py
├── visualization/
│   ├── heatmaps.py
│   ├── flow_overlay.py
│   ├── track_overlay.py
│   ├── compare_video.py
│   └── contact_sheet.py
├── report/
│   ├── html_report.py
│   ├── assets/
│   └── summary.py
├── cache/
│   ├── source_cache.py
│   ├── reference_cache.py
│   └── flow_pair_cache.py
└── validation/
    ├── metamorphic.py
    └── defect_injection.py
```

---

# 最终路线判断

按照你的真实目标，剩余工作不应该再围绕“是否要训练一个更准的总分模型”展开。

接下来最合理的主线是：

```text
自动模式与自动预算
    ↓
Cadence Integrity
    ↓
统一诊断证据
    ↓
Error Map 与 Overlay 视频
    ↓
HTML 可交互报告
    ↓
批量评测
    ↓
真实视频自动缺陷注入回归
    ↓
性能与缓存
```

完成这条路线后，项目将从当前的：

> **多模式研究型评测框架**

变成你真正需要的：

> **开箱即用、无需训练、无需为每条视频调参、能够直接给出分数并展示具体坏例的视频插帧质量诊断工具。**
