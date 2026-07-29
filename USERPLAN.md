# 核心结论

当前项目已经完成了**多模式评测框架的第一次架构闭环**，下一阶段不应继续无序增加指标，而应进入：

> **指标数学修正 → 模式能力补全 → 独立验证 → 真实数据标定 → 性能与生产化**

目前三种模式的成熟度并不一致：

| 模式               | 当前阶段   | 下一核心任务                |
| ---------------- | ------ | --------------------- |
| `no-reference`   | 可执行原型  | 修正时间尺度、补充语义时序指标并证明有效性 |
| `endpoint-2x`    | 研究可用   | 降低误报、改善语义定位并完成真实标定    |
| `full-reference` | 基础可用   | 完善全片扫描、标准 FR 指标与参考侧缓存 |
| 公共框架             | 架构基本完成 | 建立特征契约、模式预算、统一缓存与实验记录 |

项目当前已经有显式的三模式 API、独立评分 Schema 和输入契约，这是正确的基础。  但 NR 和 FR 目前主要只有冒烟测试，现有模型排序与 12 类坏例校准仍只验证 `endpoint-2x`。

---

# 一、下一版本应先冻结的产品定义

在继续开发前，需要正式冻结三个模式的分数语义。

## 1. No-Reference

输出的是：

> **Temporal Stability and Artifact Risk**

它只回答：

* 视频是否闪烁、冻结或交替模糊；
* 运动轨迹是否不平滑；
* 局部结构是否抖动；
* UI、文字或细结构是否不稳定；
* 是否存在明显技术劣化。

它不能回答：

* 生成内容是否是真实中间状态；
* 显露背景是否正确；
* 人物缺失结构是否本来就应该存在；
* 一个平滑、清晰的 hallucination 是否正确。

当前代码已经在报告中声明这一限制，并把 NR confidence 上限设为 0.75。

## 2. Endpoint-2x

输出的是：

> **Endpoint-Constrained Interpolation Quality**

参考帧提供中间时刻前后的真实端点，因此能够评价：

* 运动是否可以由两端解释；
* 中间帧是否破坏轨迹；
* 是否复制一端；
* 是否出现结构缺失、重影和错误运动层。

它仍然没有真实中间帧，所以不能等同于 Full Reference。

## 3. Full-Reference

输出的是：

> **Same-Rate Spatial and Temporal Fidelity**

它要求逐帧对应，能够直接评价空间、结构、运动和时序误差。

三个模式必须一直保持独立的：

* 特征集合；
* 特征尺度；
* 校准器；
* `score_schema`；
* 验收数据；
* 分数解释。

当前代码已经采取这个方向。

---

# 二、目标架构

建议把项目最终拆成四层。

```text
Input Contract
    ↓
Alignment / Time Grid
    ↓
Mode Executor
    ↓
Metric Registry
    ↓
Mode Fusion / Calibration
    ↓
Report / Badcase / Benchmark
```

## 第一层：输入契约

负责确定：

* mode；
* reference 是否允许或必须；
* FPS 和 cadence；
* 分辨率策略；
* 是否为同一内容；
* 是否满足发布有效分数的条件。

## 第二层：时间与对齐

分别负责：

* NR：建立真实时间网格和固定 lag；
* Endpoint：建立 source anchor 到 candidate 的 2× 单调映射；
* FR：建立 reference 到 candidate 的 1× 单调映射。

## 第三层：模式执行器

```python
evaluate_no_reference()
evaluate_endpoint_reference()
evaluate_full_reference()
```

执行器只负责编排，不应继续承载指标公式。

## 第四层：指标与融合

每个指标应返回统一结构：

```python
@dataclass
class MetricResult:
    scalars: dict[str, float]
    maps: dict[str, np.ndarray]
    instances: list[dict]
    coverage: float
    confidence: float
    warnings: list[str]
    status: str
```

这样可以解决当前“指标返回了 NaN，但窗口仍被视为有效”以及“异常被记录后融合仍可能使用残余特征”的问题。

---

# 三、Phase 0：冻结基线和可复现契约

这是下一步必须最先完成的工作。

## 0.1 建立版本契约

每份报告增加：

```json
{
  "mode": "no-reference",
  "score_schema": "nr-stability-risk-v2",
  "metric_contract": "nr-metrics-v2",
  "calibration_id": null,
  "preset_contract": "nr-standard-v1",
  "feature_contract_hash": "...",
  "backend_contract": {},
  "code_commit": "..."
}
```

当前报告只有 `score_schema` 和部分 backend 信息，还不足以严格复现实验。

## 0.2 禁止隐式改变分数

当前 NR 的：

```python
vqa_backend="auto"
```

在安装 pyIQA 的机器上会使用 NIQE，在未安装的机器上则不会使用。

这意味着同样的命令可能因为环境不同而产生不同分数。

应改为：

```text
默认：vqa_backend="none"
显式启用：vqa_backend="pyiqa-niqe"
```

启用不同 prior 时，必须使用不同的 Schema：

```text
nr-stability-risk-v2
nr-stability-risk-v2+niqe
```

NIQE 本身是自然图像统计指标，对游戏画面存在明显域偏差，只能作为极低权重的显式先验。

## 0.3 修复 coverage 计算

NR 和 FR 当前使用：

```python
sum(len(window.indices)) / n_frames
```

计算覆盖率。

窗口重叠时会重复计数。

应改为：

```python
covered = np.unique(np.concatenate([
    wf.window.indices for wf in valid_windows
]))
coverage = len(covered) / n_frames
```

置信度、报告覆盖率和 calibration 都统一使用 unique coverage。

## 0.4 明确 Python API 的 mode

当前统一入口默认 `mode=no-reference`。

为避免程序静默进入错误模式，建议：

```python
def evaluate(..., mode: EvaluationMode | None = None):
    if mode is None:
        raise ValueError("mode must be explicitly specified")
```

旧的 `evaluate_vfi()` 继续承担 Endpoint 兼容入口即可。

## Phase 0 验收

* 相同输入和相同显式 backend，跨机器报告特征定义一致；
* 可选依赖不会隐式改变 Schema；
* mode 不允许隐式推断；
* 报告可以定位到代码、权重、配置和校准器；
* coverage 不重复计数。

---

# 四、Phase 1：公共执行引擎重构

当前三个执行器已经分开，但仍有大量重复编排代码。下一步应把公共流程抽离。

## 1.1 模式专属 Preset

目前所有模式共用同一个 `Preset`，其中很多字段只对 Endpoint 有意义，例如：

* `run_region_branches`；
* `audit_top_fraction`；
* `full_res_edges`；
* `run_tracker`；
* `run_depth`。

建议定义：

```python
@dataclass
class ExecutionBudget:
    scan_width: int
    flow_width: int
    uniform_windows: int
    risk_windows: int
    native_audit_windows: int
    max_flow_pairs: int
```

再为模式分别构造：

```python
get_budget("no-reference", "standard")
get_budget("endpoint-2x", "standard")
get_budget("full-reference", "standard")
```

例如：

* NR Audit：增加长时间窗口和轨迹跟踪；
* Endpoint Audit：原分辨率 source/candidate 重算；
* FR Audit：原分辨率感知指标和局部运动误差。

## 1.2 建立特征注册表

新增：

```text
rr_vfiqa/fusion/feature_registry.py
```

每个特征定义：

```python
FeatureDefinition(
    name="nr_mct_short_mean",
    mode="no-reference",
    category="temporal",
    units="luma",
    direction="lower_is_better",
    resolution_invariant=True,
    required=True,
    scale=...,
    version="v2",
)
```

这样可以避免：

* 特征名字与实际含义不一致；
* 像素尺度特征在 960 和 native resolution 下混合；
* required feature 只靠字符串硬编码；
* 校准器 feature order 与执行器脱节。

## 1.3 建立统一 Flow Pair Planner

NR 的 120 FPS 窗口通常包含约 9 帧。`WindowFlows.precompute()` 默认只预计算帧距不超过 2 的 pair，而 1/30 秒 lag 在 120 FPS 下跨越约 4 帧，后续会逐对补算。

应让每个 metric 先声明需要的 pair：

```python
flow_plan = FlowPairPlan()

flow_plan.add(a, b, reason="self-composition")
flow_plan.add(a, m, reason="self-composition")
flow_plan.add(m, b, reason="self-composition")
flow_plan.add(short_a, short_b, reason="mct-short")
flow_plan.add(medium_a, medium_b, reason="mct-medium")
```

然后：

```python
flows.precompute(flow_plan.unique_pairs())
```

一次批量运行 RAFT，避免中途逐对推理。

## 1.4 通用视频 FlowStore

新增：

```text
rr_vfiqa/cache/video_flow_store.py
```

缓存 key 包含：

* video hash；
* frame pair；
* flow width；
* backend；
* weights；
  -算法版本。

用途：

* Endpoint：继续缓存 source；
* FR：多个候选共享 reference flow；
* NR：同一视频重复报告或 Audit 重用；
* 不同窗口重复 pair 时避免重复推理。

---

# 五、Phase 2：No-Reference V2

这是当前最大的新功能缺口。

## 2.1 统一自参考的真实时间尺度

当前每个相邻三帧都作为虚拟：

```text
endpoint / middle / endpoint
```

但这意味着：

| 输入      |   两端跨度 |
| ------- | -----: |
| 60 FPS  | 1/30 秒 |
| 120 FPS | 1/60 秒 |

因此当前 `nr_self_comp_mean` 和 `nr_self_cycle_mean` 在 60 与 120 FPS 下不是同一物理尺度。

应建立统一的：

```python
TemporalTriplet(
    left_time=t - 1/60,
    middle_time=t,
    right_time=t + 1/60,
)
```

即两端总跨度固定为 1/30 秒。

* 60 FPS：使用连续 3 帧；
* 120 FPS：使用间隔 2 帧的 3 帧；
* VFR：按 PTS 选择最接近时刻。

同时保留 native triplet，但作为独立高频特征：

```text
nr_native_self_comp
nr_common_self_comp
```

## 2.2 修复 acceleration 和 jerk

当前从 1/60 秒跨度的 flow 计算 speed 后，固定使用：

```python
dt = 1 / 60
```

求 acceleration 和 jerk。

120 FPS 下，相邻 speed 样本的中心时间通常是 1/120 秒，所以当前导数尺度错误。

正确实现：

```python
flow_times = 0.5 * (times[a] + times[b])
velocity = displacement / (times[b] - times[a])

acceleration = np.gradient(velocity, flow_times)
jerk = np.gradient(acceleration, flow_times)
```

同时不要只统计全图 median flow。建议加入网格级统计：

```text
4×4 或 8×8 tile velocity
tile acceleration P90
tile jerk P90
local reversal fraction
```

这样才能发现：

* 剑尖抖动；
* 人物肢体反向；
* 柱子和背景运动分层错误。

## 2.3 三档 Lag

NR 必须同时计算：

```text
native cadence
1/60 秒
1/30 秒
```

原因是：

* 120 FPS 单个生成帧复制只能在 1/120 秒邻接中直接发现；
* 1/60 秒用于跨 60/120 的共同尺度；
* 1/30 秒用于中尺度轨迹和运动补偿。

建议特征：

```text
nr_duplicate_native
nr_freeze_native
nr_mct_native
nr_mct_1_60
nr_mct_1_30
nr_edge_instability_native
nr_edge_instability_1_60
```

当前只使用 1/60 秒检测 duplicate/freeze，可能漏掉 120 FPS 中单个生成帧复制。

## 2.4 遮挡感知

当前 NR composition 没有传入 confidence/occlusion，self-cycle 也主要依赖 splat coverage。

需要为每个虚拟端点 pair 计算：

```python
occ = cycle_occlusion(f_ab, f_ba)
weight_a = occ.conf_ab * (1 - occ.occ_ab)
weight_b = occ.conf_ba * (1 - occ.occ_ba)
```

然后：

* composition 使用方向对应的网格权重；
* reconstruction 使用 forward-splat 后的 visibility；
* 单独输出 disocclusion risk，不让它污染可见区域主分。

## 2.5 NR 区域时序分支

Endpoint 的语义分支不能原样迁移，但可以迁移“时序稳定性”部分。

### UI / 文字

从 candidate 自身建立 screen-static mask：

* 多帧位置稳定；
* 持久边缘；
* HUD 位置先验；
* 排除大面积静态背景。

评测：

* native-lag edge XOR；
* gradient fluctuation；
* component count fluctuation；
* stroke persistence；
* screen-space subpixel drift。

### 细物体

使用 LSD/edge skeleton 检测候选细结构：

* 跨帧关联；
* 长度变化；
* 角度二阶变化；
* endpoint 断裂；
* 背景相对运动变化。

NR 下不能判断“它应该跟哪一层运动”，但可以判断“它是否突然改变运动层或发生闪烁”。

### 人物与武器

使用 proxy mask 或可选分割模型：

* 角色轮廓面积变化；
* connected components 跳变；
* body-relative track jerk；
* 轮廓双边缘；
* 局部结构频闪。

报告必须使用：

```text
character_temporal_risk
thin_structure_temporal_risk
appendage_track_risk
```

不能使用 `character_missing` 这种具有真实性含义的标签。

## 2.6 NR Compare 输入保护

NR compare 必须验证：

* FPS 属于同一档；
* 时长接近；
* 分辨率或 aspect 一致；
* 场景 fingerprint 相似；
* PTS 时间线相近。

不满足时：

```text
拒绝排序
```

或要求显式：

```bash
--allow-cross-content
```

并把输出改为“独立质量报告”，不能提供 `relative_vs_mean`。

## NR V2 验收

建立至少以下测试：

| 测试                              | 预期                            |
| ------------------------------- | ----------------------------- |
| 120 FPS clean vs odd-frame copy | copy 的 duplicate/temporal 分下降 |
| 60 FPS clean vs freeze          | freeze 被定位                    |
| 60/120 同一内容重采样                  | 共同时间尺度特征接近                    |
| blur severity 递增                | temporal/phase 分单调下降          |
| rotation tear                   | motion risk 上升                |
| UI drift                        | UI stability 下降               |
| sword flicker                   | local track jerk 上升           |
| scene cut                       | 不作为插帧缺陷评分                     |
| smooth hallucination            | 报告保持低 confidence，不声称正确        |

---

# 六、Phase 3：Full-Reference V2

## 3.1 全片低分辨率参考扫描

当前 FR 窗口仍主要由 candidate 自身风险和均匀采样决定。

这会漏掉：

* 全程稳定模糊；
* 始终错误的 UI；
* 稳定缺失对象；
* 全程颜色偏差；
* 稳定错误纹理。

新增：

```text
sampling/full_reference_scan.py
```

对全部匹配帧低分辨率计算：

```text
Y L1
chroma L1
gradient L1
edge mismatch
local SSIM proxy
frame-difference mismatch
```

窗口风险应合成：

```python
risk = (
    candidate_self_risk
    + reference_spatial_error
    + reference_temporal_error
)
```

全片扫描结果还应直接进入全局分数，而不是只负责采样。

## 3.2 明确分辨率策略

当前 alignment 只检查宽高比，而 metric 要求完全相同 shape。

新增：

```python
geometry_policy:
    strict
    resize-candidate
    common-resolution
```

生产默认建议：

```text
strict
```

只允许相同宽高。其他策略必须在报告中记录 resize transform，并使用不同 score schema。

## 3.3 修正指标名称与实现

当前 `_ssim()` 是全图统计，不是标准局部 SSIM。

当前 `fr_multiscale_perceptual` 实际是多尺度 Luma L1。

应修改为：

```text
fr_global_ssim_proxy           删除或仅保留诊断
fr_ssim                        实现 11×11 Gaussian local SSIM
fr_ms_ssim                     可选
fr_multiscale_luma_l1          对现有指标正确命名
fr_lpips                       可选
fr_dists                       可选
```

不要将普通多尺度 L1 命名为 perceptual。

## 3.4 局部运动与时序指标

当前 trajectory 使用全图 median flow 累积，容易被相机运动主导。

改为：

* tile-wise median flow；
* 前景 ROI flow；
* edge ROI flow；
* track trajectories；
* P50/P90/P99 局部误差；
* camera residual flow error。

结构持续性也应先运动补偿，再比较 edge birth/death，避免正常移动造成 XOR。

## 3.5 参考侧缓存

多个候选使用同一个 reference 时，应缓存：

* reference frames at working width；
* reference flow；
* reference edges；
* reference tile tracks；
* reference temporal features。

成本变成：

[
T_\text{reference}+N\cdot T_\text{candidate}
]

目前 FR 每个候选都会重新计算 reference flow。

## FR V2 验收

* reference 与自身得到满分；
* blur、颜色漂移、结构删除、位移、冻结各自降低正确子分；
* 全程稳定模糊可被全片扫描发现；
* 局部对象删除不会被背景平均淹没；
* 不同分辨率根据 policy 明确拒绝或处理；
* unrelated capture 必须 fail-closed；
* 与标准 PSNR、SSIM、LPIPS 和人工评分建立相关性结果。

---

# 七、Phase 4：Endpoint 精度提升

Endpoint 当前最大问题不是缺少指标，而是误报率和标签解释。

严格 12 类合成定位当前只有：

* Recall 0.667；
* Precision 0.229；
* F1 0.340。

## 4.1 只用有效窗口融合

当前已经计算 `valid_wfs`，但 category aggregation 仍使用全部 `wfs`。

应按类别定义有效性：

```text
motion_valid_windows
temporal_valid_windows
structure_valid_windows
semantic_valid_windows
```

某个阶段失败的窗口不能用剩余少量特征形成偏乐观类别分。

## 4.2 Audit 特征做分辨率归一化

960 宽和 native resolution 的：

* Chamfer 像素；
* flow residual；
* component size；
* line width；

不能直接使用同一阈值。

所有空间尺度特征应改为：

```text
除以图像对角线
除以目标尺度
除以 endpoint flow magnitude
```

或为 Audit 特征使用独立名字与归一化器。

## 4.3 分离证据和语义标签

报告不要直接从一个启发式特征推出确定语义。

改为：

```json
{
  "evidence": [
    "edge_support_loss",
    "local_track_jerk",
    "foreground_mask_area_drop"
  ],
  "inferred_types": [
    {
      "type": "character_missing",
      "confidence": 0.43
    }
  ]
}
```

低 precision 类型默认只显示 evidence，不输出确定语义标签。

## 4.4 每类独立阈值校准

对 12 类坏例输出：

* confusion matrix；
* per-class precision/recall/F1；
* threshold curve；
* backend 分层结果；
* Fast/Standard/Audit 分层结果。

优先改善：

1. `head_erase`；
2. `pole_wrong_motion`；
3. `sword_flicker`；
4. `ui_drift`；
5. `text_merge`；
6. `shop_jump`。

不要用一个统一 category threshold 覆盖全部类型。

---

# 八、Phase 5：独立校准体系

三个模式必须拥有三套数据与校准工具。

## 5.1 NR 数据

即使推理时无参考，训练和验证时仍可以利用真实高帧率 GT。

数据构造：

```text
真实 120/240 FPS
    ↓
降采样生成 source
    ↓
多个真实插帧模型
    ↓
得到 60/120 候选
```

评测器推理只看 candidate，但标定目标可以来自：

* 真实中间帧误差；
* 人工 artifact severity；
* A/B preference；
* 局部坏例时间段。

NR 主要目标不是 MOS，而是：

```text
同源候选排序
artifact risk
badcase localization
```

## 5.2 Endpoint 数据

继续使用：

* 真实高帧率伪 GT；
* 不同插帧模型；
* 不同游戏；
* 不同编码；
* 人工 A/B；
* artifact labels。

## 5.3 FR 数据

使用逐帧 GT 构造：

* blur；
* color shift；
* spatial shift；
* local deletion；
* ghost；
* freeze；
* temporal jitter；
* codec variants；
* 真实模型输出。

## 5.4 数据划分

必须按以下维度隔离：

```text
game
scene
interpolation model
motion type
capture pipeline
codec
```

报告：

* Leave-One-Game-Out；
* Leave-One-Model-Out；
* SRCC；
* PLCC；
* pairwise accuracy；
* per-class AP/F1；
* worst-10% recall；
* calibration error；
* runtime 与 VRAM。

建议验收门槛作为开发目标：

| 模式                           |                    初始目标 |
| ---------------------------- | ----------------------: |
| NR 同源 A/B accuracy           |                   ≥ 80% |
| NR artifact worst-10% recall |                   ≥ 85% |
| Endpoint 同源 A/B accuracy     |                   ≥ 85% |
| Endpoint 12 类 precision      | ≥ 50%，同时保持 recall ≥ 65% |
| FR 与人工质量 SRCC                |                  ≥ 0.85 |
| FR 同源排序 accuracy             |                   ≥ 90% |

在达到真实数据门槛前，报告必须继续显示：

```text
calibrator = formula
production_gate = false
```

---

# 九、Phase 6：实验记录和文档路线

建议建立以下文档。

```text
docs/
├── architecture/
│   ├── MODE_CONTRACTS.md
│   ├── EXECUTION_ENGINE.md
│   └── CACHE_CONTRACT.md
├── metrics/
│   ├── NR_V2.md
│   ├── ENDPOINT_V2.md
│   └── FR_V2.md
├── validation/
│   ├── DATASET_PROTOCOL.md
│   ├── HUMAN_RATING_PROTOCOL.md
│   ├── SPLIT_POLICY.md
│   └── ACCEPTANCE_GATES.md
├── adr/
│   ├── 001-explicit-modes.md
│   ├── 002-independent-score-schemas.md
│   ├── 003-no-cross-mode-ranking.md
│   └── 004-formula-vs-calibrated-score.md
└── ROADMAP.md
```

每次实验生成：

```text
runs/<run_id>/
├── config.json
├── provenance.json
├── report.json
├── runtime.json
├── windows.jsonl
├── features.jsonl
├── badcases/
├── heatmaps/
└── logs.txt
```

`provenance.json` 至少记录：

* git commit；
* dirty state；
* Python 与依赖版本；
* GPU/NPU；
* backend 和权重；
* score schema；
* calibration ID；
* dataset manifest hash；
  -运行命令；
  -随机种子。

项目已经有基础 `calibration_provenance()`，可以在此基础上扩展。

校准器目录建议：

```text
calibrators/<score_schema>/<calibration_id>/
├── model.pkl
├── feature_order.json
├── training_manifest.json
├── validation_metrics.json
├── thresholds.json
└── model_card.md
```

---

# 十、Phase 7：性能与部署

在指标稳定前不要过早优化，但应提前建立性能可观测性。

## 必须记录的分阶段耗时

```text
decode
cheap_scan
alignment
flow
region_metrics
fusion
report
```

## 优化顺序

1. 共享低分辨率顺序解码；
2. Flow Pair Planner；
3. RAFT 批处理；
4. 跨窗口 pair cache；
5. FR reference cache；
6. Audit 仅升级最高风险窗口；
7. mixed precision；
8. CUDA/NPU 后端适配。

## NPU

当前代码中的 `device="npu"` 并不代表真实 NPU 支持；RAFT 会在非 CUDA 环境回退 CPU。

应选择其中一种：

* 暂时从公开支持列表删除 NPU；
* 或增加独立 `AscendFlowBackend`，完成 torch_npu/ONNX/ACL 的明确实现和测试。

不要保留“参数可填，但实际回退 CPU”的模糊状态。

---

# 十一、推荐执行顺序

严格按照以下依赖顺序推进：

```text
Phase 0  可复现契约和 P0 数学问题
    ↓
Phase 1  公共执行引擎、Feature Registry、Flow Planner
    ↓
Phase 2  No-Reference V2
    ↓
Phase 3  Full-Reference V2
    ↓
Phase 4  Endpoint 误报与语义精度
    ↓
Phase 5  三模式独立真实标定
    ↓
Phase 6  文档、实验记录、模型卡
    ↓
Phase 7  性能、CUDA/NPU 与生产部署
```

实际开发中验证测试应跟随每个 Phase 同步完成，不要最后集中补测试。

---

# 十二、近期应直接创建的任务清单

第一批任务应限制在这些内容：

1. 新建 Feature Registry 和 MetricResult 数据结构。
2. 修复 unique coverage。
3. 将 `evaluate()` 改为强制显式 mode。
4. 将 NR 默认 VQA 改为 `none`。
5. 新建 TemporalLagPlan。
6. 修复 NR acceleration/jerk 时间间隔。
7. 增加 native、1/60、1/30 三档 lag。
8. 将 NR self-reference 统一到固定物理跨度。
9. 为 NR composition/cycle 增加遮挡权重。
10. 增加 NR compare 内容和 FPS 一致性校验。
11. 明确 FR geometry policy。
12. 新增 FR 全片低分辨率 reference scan。
13. 将伪 SSIM 和伪 perceptual 指标正确改名或替换。
14. Endpoint 融合改用类别有效窗口。
15. 建立 NR 与 FR 独立 validation harness。
16. 更新 CI，加入三模式方向性测试。

---

# 最终路线判断

当前项目不需要再次推翻重构。正确策略是保留已经完成的三模式骨架，然后重点解决：

> **NR 的时间尺度与能力效度、FR 的全片参考利用、Endpoint 的误报率，以及三套互不混用的真实标定。**

下一版本不应以“增加了多少指标”为成功标准，而应以以下四件事为准：

1. 数学定义在 60/120 FPS 下成立；
2. 坏例出现时正确子分稳定下降；
3. 最差窗口能准确落在真实坏例时间段；
4. 在未见过的游戏和模型上仍能保持排序能力。
