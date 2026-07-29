# rr_vfiqa — 多模式视频插帧质量评测

`rr_vfiqa` 是面向游戏视频插帧的研究型评测框架。项目现在明确区分三种数学条件不同的评测模式，不会再把任意输入都套入 60→120 FPS 的奇偶帧假设。

| 模式 | 输入 | 输出含义 | 可信度定位 |
|---|---|---|---|
| `no-reference` | 单条 60 或 120 FPS 视频 | 时序稳定性与伪影风险 | 最低，不代表真实插帧误差 |
| `endpoint-2x` | 60 FPS 端点参考 + 120 FPS 候选 | 端点约束下的插帧质量 | 中等，是原有成熟内核 |
| `full-reference` | 逐帧对应的 60 FPS GT + 60 FPS 候选 | 同帧空间与时序保真度 | 最高 |

三种模式具有独立的输入契约、特征集合、融合权重、子分数和 `score_schema`。不同模式的 `overall_score` 不在同一标尺上，禁止跨模式直接排序。

> 当前状态仍是 research prototype / metric development framework。三种模式都使用未完成真实数据标定的公式融合，不能仅凭 `overall_score` 作为生产上线门禁。

---

## 安装

```bash
pip install -e .            # NumPy、OpenCV、SciPy、PyAV
pip install -e ".[torch]"   # RAFT 光流，推荐 GPU 环境使用
pip install -e ".[fusion]"  # Endpoint 模式的 LightGBM 校准器
pip install -e ".[vqa]"     # NR 模式的可选 pyIQA/NIQE 弱先验
pip install -e ".[dev]"     # pytest
```

NR 默认使用 `--vqa-backend none`，因此基础 schema 不受本机可选依赖影响。若明确启用 `--vqa-backend pyiqa-niqe`，报告 schema 会增加 `+niqe` 后缀，并记录该后端契约；不再提供会随环境变化的 `auto` 行为。

---

## 快速使用

推荐使用统一的显式入口：

```python
from rr_vfiqa import evaluate

# 单视频无参考：60/120 FPS
nr_report = evaluate(
    candidate_video="video_120.mp4",
    mode="no-reference",
    preset="standard",
)

# 60→120 端点参考
endpoint_report = evaluate(
    reference_video="source_60.mp4",
    candidate_video="output_120.mp4",
    mode="endpoint-2x",
    preset="standard",
)

# 60→60 同帧完整参考
fr_report = evaluate(
    reference_video="ground_truth_60.mp4",
    candidate_video="output_60.mp4",
    mode="full-reference",
    preset="standard",
)
```

三个独立执行器也可直接调用：

```python
from rr_vfiqa import (
    evaluate_no_reference,
    evaluate_endpoint_reference,
    evaluate_full_reference,
)
```

旧的 `evaluate_vfi(source_video, candidate_video, ...)` 仍保留，语义固定为 `endpoint-2x`，用于兼容已有调用方。新代码应使用 `evaluate()` 或独立执行器，让评测假设在调用点可见。

CLI 必须显式指定模式：

```bash
# 单视频无参考
rr-vfiqa evaluate \
  --mode no-reference \
  --candidate video_120.mp4 \
  --out runs/nr

# 60 FPS 端点参考 + 120 FPS 插帧
rr-vfiqa evaluate \
  --mode endpoint-2x \
  --reference source_60.mp4 \
  --candidate output_120.mp4 \
  --out runs/endpoint

# 逐帧对应的 60 FPS GT + 60 FPS 输出
rr-vfiqa evaluate \
  --mode full-reference \
  --reference ground_truth_60.mp4 \
  --candidate output_60.mp4 \
  --out runs/fr
```

输入契约会主动拒绝歧义：

- `no-reference` 不接受 `--reference`；
- `endpoint-2x` 和 `full-reference` 必须提供 `--reference`；
- `full-reference` 遇到非 1× 帧率、几何不一致或低覆盖对齐时 fail-closed；
- `endpoint-2x` 遇到不可靠的 2× 锚点对齐时 fail-closed。

`full-reference` 默认使用 `--geometry-policy strict`，要求原始宽高完全一致。`resize-candidate` 会显式把候选变换到参考工作网格，`common-resolution` 则把两路降到共同工作分辨率；两者都要求宽高比一致，并分别使用 `+resize-candidate`、`+common-resolution` schema 后缀，报告也会保存 resize transform。

同一模式内可以比较多个候选：

```bash
rr-vfiqa compare \
  --mode endpoint-2x \
  --reference source_60.mp4 \
  --candidates model_a.mp4 model_b.mp4 \
  --out runs/compare
```

---

## 模式一：No-Reference 60/120 FPS

### 评测内容

NR 执行器按 PTS 构造原生跨度与固定物理跨度的虚拟端点参考：

```text
相位 0：Y0 / Y2 / Y4 ... 为虚拟端点，Y1 / Y3 ... 为中间帧
相位 1：Y1 / Y3 / Y5 ... 为虚拟端点，Y2 / Y4 ... 为中间帧
```

两个相位只作为对称假设使用，框架不会声称某一相位是真实帧。核心证据包括：

- 原生相邻三元组与固定 ±1/60 秒三元组的 flow composition；
- 使用前后向一致性遮挡与置信度加权的 self-cycle 残差；
- native、1/60 秒与 1/30 秒三档运动补偿残差；
- flow velocity、acceleration 和 jerk 风险；
- 去除全局平移后的 flow Jacobian、folding、divergence 和 curl；
- 相机相对 KLT 点轨迹 acceleration、jerk 与方向突变；
- 与相位标签无关的清晰度、边缘交替；
- 统一时间尺度及 120 FPS 原生相邻帧的重复、冻结检测；
- HUD 边缘与文字密集笔画的屏幕坐标不稳定；
- 全局模糊、噪声、块效应；
- 可选 pyIQA/NIQE 弱先验，融合权重不超过 5%。

窗口按 PTS 和真实时间跨度选择。±33.3 ms 的窗口在 60 FPS 下通常包含 5 帧，在 120 FPS 下通常包含 9 帧，因此 120 FPS 的额外高频信息不会被固定“五帧窗口”丢掉。velocity、acceleration 和 jerk 全部使用真实时间间隔计算，而不是把不同 FPS 的帧索引差当成相同时间。

### 报告语义

NR 报告使用：

```text
meta.mode = "no-reference"
meta.score_schema = "nr-stability-risk-v2"
meta.score_semantics = "temporal stability and artifact risk; not interpolation truth"
```

子分数为：

- `temporal_stability`
- `motion_smoothness`
- `phase_consistency`
- `ui_text_stability`
- `technical_quality_prior`

NR 无法证明真实轨迹、显露背景或清晰 hallucination 是否正确，所以 confidence 上限为 0.75。非 60/120 FPS 输入目前会 fail-closed，而不是套用未经标定的尺度。多个 NR 候选只有在 FPS 桶、时长、画幅和抽样内容指纹一致时才能排序；`--allow-cross-content` 只会生成互相独立的报告，不发布相对排名。

---

## 模式二：Endpoint-Referenced 60→120

这是项目原有且最成熟的模式。参考视频提供每个生成中间帧前后的真实端点，但不提供真实中间帧。

核心能力包括：

| 指标 | 主要用途 |
|---|---|
| 双向 Endpoint flow composition | 撕裂、错误运动层、错误运动路径 |
| Reverse anchor cycle | 时序不一致、系统性生成帧模糊 |
| 运动补偿时序残差 | 拖影、闪烁、边缘摆动 |
| 已知奇偶相位检测 | 生成帧清晰度或结构交替 |
| Endpoint edge support | 结构缺失、重影轮廓 |
| UI、文字、转场、卡牌代理 | HUD 漂移、笔画粘连、双重曝光 |
| 人物、细物体、武器代理 | 局部缺失、错误归属、轨迹抖动 |

该模式保留内容哈希缓存、局部 drop/duplicate 恢复、原分辨率 Audit 升级和错误阶段 fail-closed。类别融合只使用该类别所依赖阶段均成功的窗口；某个窗口的 temporal 阶段失败，不会被其他残留特征当作有效 temporal 证据。报告使用：

```text
meta.mode = "endpoint-2x"
meta.score_schema = "endpoint-reduced-reference-v1"
```

人物、细物体、UI 等经典启发式仍会在 `meta.proxy_branches` 中声明，不能解释成语义真值。

---

## 模式三：Full-Reference Same-Rate

该模式要求 reference 与 candidate 来自同一次录制并逐帧对应。执行器先进行 1× PTS + 低分辨率内容单调对齐，再对完整时间线执行低分辨率 reference scan；全片 Y/chroma、梯度、边缘、局部 SSIM proxy 与帧差不一致风险会参与窗口采样和融合。随后在匹配窗口中计算：

- Y/RGB L1、RGB Charbonnier、PSNR、11×11 Gaussian local SSIM；
- 明确命名的多尺度亮度与梯度 L1（不是 perceptual metric）；
- 边缘 precision、recall、F1 与 Chamfer；
- UI、文字、显著结构和中心运动区域的局部参考误差；
- reference/candidate 帧间变化误差；
- 光流差异与轨迹偏差；
- 运动补偿残差差异；
- flicker excess；
- 结构持续性误差。

它不会复用 Endpoint 模式的 anchor/generated 奇偶定义、半程 freeze-copy 公式或 `pair_of_candidate`。报告使用：

```text
meta.mode = "full-reference"
meta.score_schema = "fr-same-rate-fidelity-v2"
```

子分数为 `spatial_fidelity`、`structural_fidelity`、`temporal_fidelity` 和 `motion_fidelity`。

如果两条 60 FPS 视频只是同场景但不是逐帧对应，当前模式会拒绝发布有效分数。局部 DTW、时间伸缩、crop 和颜色归一化不属于当前 same-rate 契约；跨分辨率仅由显式的 `resize-candidate` 或 `common-resolution` policy 支持。

---

## 报告与失败语义

三种模式统一输出 `Report`：

```json
{
  "overall_score": 82.4,
  "confidence": 0.71,
  "scores": {},
  "event_ambiguity": 0.0,
  "worst_windows": [],
  "features": {},
  "meta": {
    "mode": "no-reference",
    "score_schema": "nr-stability-risk-v2",
    "status": "ok",
    "metric_contract": "nr-metrics-v2",
    "preset_contract": "nr-standard-v1",
    "feature_contract_hash": "...",
    "backend_contract": {},
    "code_commit": "...",
    "working_tree_dirty": false,
    "production_gate": false
  }
}
```

启用 `--out` 后生成：

- `report.json`
- `timeline.md`
- 可用 Matplotlib 时生成 `timeline.png`
- 启用坏例导出时生成 `badcases/*.mp4`

核心特征缺失、输入契约不成立或对齐不可靠时：

- `overall_score` 序列化为 `null`；
- `meta.status` 为 `failed`；
- confidence 降到接近零；
- CLI 返回非零退出码；
- 阶段错误保存在 `meta.stage_errors`。

coverage 统一按所有有效窗口覆盖到的唯一帧集合计算，重叠窗口不会重复计数。所有模式的报告均记录 metric、preset、feature、backend、代码提交与工作区状态契约，并固定声明 `production_gate=false`，直到各模式的真实数据标定独立完成。

---

## Preset

| Preset | 扫描宽度 | 窗口数量 | 光流宽度 | Endpoint 区域分支 | Endpoint Audit |
|---|---:|---:|---:|---|---|
| `fast` | 320 | 8+8 | 480 | 关闭 | 无 |
| `standard` | 384 | 16+32 | 960 | 轻量 | 无 |
| `audit` | 480 | 24+48 | 960 | 完整 | 最高风险 10%，最多 64 个 |

NR/FR 使用相同的资源预算，但使用 PTS 时间窗口和各自的指标执行器。Endpoint 的 Audit 会在原分辨率重新构建 candidate/source flow、遮挡、camera、edge 和人物 ROI。

---

## 校准、测试与基准

Endpoint 原有的合成标定与基准入口继续保留：

```bash
python -m rr_vfiqa.calibration.validate --synthetic --workdir cal_run
python -m rr_vfiqa.calibration.model_validation --workdir mv_run
python -m rr_vfiqa.calibration.detection_eval --workdir det_run
python -m rr_vfiqa.benchmark --source src.mp4 --candidates a.mp4 b.mp4
```

现有 Endpoint 合成证据不能迁移成 NR 或 FR 的校准证据。三种模式需要分别收集真实样本、人工排序与阈值：

1. NR 60/120 的真实伪影与稳定性标注；
2. Endpoint 的真实 120/240 FPS 伪 GT 和人工 A/B；
3. FR 60→60 的逐帧 GT、真实模型输出与场景外验证。

NR 与 FR 的方向性验证使用独立 manifest，不共享 score schema 或阈值：

```bash
python -m rr_vfiqa.calibration.mode_validation \
  --manifest validation/nr_manifest.json \
  --output validation/nr_metrics.json

python -m rr_vfiqa.calibration.mode_validation \
  --manifest validation/fr_manifest.json \
  --output validation/fr_metrics.json
```

manifest 顶层指定 `mode`，每个 case 提供 `better`、`worse`；FR case 额外提供 `reference`。该 harness 只报告模式内方向性准确率并保持 `production_gate=false`。

运行测试：

```bash
python -m pytest
RR_VFIQA_TEST_FLOW=raft python -m pytest
```

测试覆盖旧 Endpoint 回归、模式输入契约、NR 60/120 时间窗口、FR 同帧完美匹配与冻结劣化、对齐 fail-closed、缓存隔离、warp 方向、校准器和合成坏例。

---

## 目录结构

```text
rr_vfiqa/
├── multimode.py                 统一分发、NR/FR 执行器、同模式比较
├── pipeline.py                  Endpoint-2x 执行器与旧 API 兼容层
├── config.py                    EvaluationMode 与模式输入契约
├── schema.py                    对齐、窗口、报告和 warp 数据结构
├── io/
│   ├── video_reader.py          PyAV + PTS 解码
│   ├── timestamp_alignment.py   Endpoint 2× 对齐
│   └── full_reference_alignment.py  Full-Reference 1× 对齐
├── sampling/
│   ├── window_selector.py       Endpoint 生成帧窗口
│   ├── time_window_selector.py  NR/FR 时间跨度窗口
│   ├── temporal_plan.py         NR 的 PTS lag/triplet/flow 规划
│   └── full_reference_scan.py   FR 全时间线低分辨率参考扫描
├── metrics/
│   ├── no_reference.py          双相位自参考与通用时序指标
│   ├── full_reference.py        空间/时序同帧参考指标
│   └── ...                      Endpoint 原有指标
├── fusion/
│   ├── score_schema.py          Endpoint 融合
│   ├── mode_score_schemas.py    NR/FR 独立融合
│   └── feature_registry.py      版本化特征、方向、单位与必需项
├── cache/                       Endpoint source 特征缓存
├── motion/                      RAFT/Farneback、warp、遮挡、camera
├── regions/                     Endpoint UI/文字/人物/细物体代理
├── models/                      flow/tracker/depth/VQA 后端
├── report/                      JSON、timeline、badcase
├── testing/                     合成场景与缺陷语料
└── calibration/                 各模式独立验证与 Endpoint 标定工具
```

最重要的使用原则只有一个：先确认手中的 reference 到底是“端点参考”还是“逐帧 Ground Truth”，再选择模式；不要让程序自动猜测 60+60 两条视频的语义。
