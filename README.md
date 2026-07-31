# rr_vfiqa

> 接收一条候选视频以及可选参考视频，自动判断能够安全使用的评测条件，输出质量分数、置信度、问题时间段、热力图、关键帧和坏例视频。

```
不要求用户手动选择 NR / Endpoint / FR
不能安全判断时 fail-closed，不发布伪分数
分数是工程风险或保真度，不是人类主观 MOS
```

| 模式 | 输入 | 输出含义 | 可信度 |
|---|---|---|---|
| `no-reference` | 单条 60 或 120 FPS 视频 | 时序稳定性与伪影风险 | 最低 |
| `endpoint-2x` | 60 FPS 端点参考 + 120 FPS 候选 | 端点约束下的插帧质量 | 中等 |
| `full-reference` | 逐帧对应的 60 FPS GT + 60 FPS 候选 | 同帧空间与时序保真度 | 最高 |

三种模式的 `overall_score` 不在同一标尺上，**禁止跨模式直接比较**。

> **项目状态：** research prototype / metric development framework。公式融合未完成真实数据标定，`production_gate=false`，不能仅凭 `overall_score` 作为生产上线门禁。

---

## 60 秒快速开始

```bash
pip install -e ".[torch]"
```

只有候选视频：

```bash
rr-vfiqa inspect --candidate output.mp4 --out runs/nr
```

候选 + 参考（自动判断模式）：

```bash
rr-vfiqa inspect \
  --candidate output.mp4 \
  --reference source.mp4 \
  --out runs/example
```

`inspect` 自动完成：**模式路由 → 评分 → 诊断 → HTML 报告与坏例视频**。

---

## 我该提供哪些输入

| 你拥有的文件 | 自动模式 | 可以回答的问题 |
|---|---|---|
| 只有候选视频 | No-Reference | 是否存在冻结、交替、抖动、模糊、UI 不稳定等风险 |
| 原始低帧率端点 + 2× 插帧结果 | Endpoint-Referenced | 中间帧是否符合端点运动和结构约束 |
| 同帧率逐帧 GT + 候选 | Full-Reference | 候选与真实目标的空间、时序和运动保真度 |

```text
只有 candidate           → No-Reference
candidate + reference     → FPS≈2×  → Endpoint-Referenced
                          → FPS相同 → Full-Reference
无法安全满足任何契约      → fail-closed，生成 failed 报告，不猜测模式
```

---

## 输出结果是什么

```text
runs/example/
├── report.json          # 完整结构化结果
├── report.html          # 交互式诊断报告
├── timeline.md          # 时间线文本
├── timeline.png         # 时间线图
├── heatmaps/            # 问题空间热力图
│   └── issue_000_*.png
└── badcases/            # 坏例视频与关键帧
    ├── issue_000_original.mp4
    ├── issue_000_overlay.mp4
    ├── issue_000_compare.mp4
    └── issue_000_keyframe.png
```

[打开完整交互教程](docs/index.html)

---

## 安装

```bash
pip install -e .            # NumPy、OpenCV、SciPy、PyAV
pip install -e ".[torch]"   # RAFT 光流（推荐 GPU）
pip install -e ".[fusion]"  # Endpoint LightGBM 校准器
pip install -e ".[vqa]"     # NR 可选 pyIQA/NIQE 弱先验
pip install -e ".[dev]"     # pytest
```

---

## 常用工作流

### 单视频自动评测

```bash
rr-vfiqa inspect --candidate output.mp4 --out runs/nr
```

### 批量评测

```bash
rr-vfiqa inspect-batch --manifest jobs.json --out runs/batch
```

```json
{ "items": [
  { "id": "model_a", "candidate": "a.mp4", "reference": "ref.mp4" },
  { "id": "single_b", "candidate": "b.mp4" }
] }
```

### 显式模式（高级用户）

```bash
rr-vfiqa evaluate --mode no-reference     --candidate video.mp4 --out runs/nr
rr-vfiqa evaluate --mode endpoint-2x      --reference src.mp4 --candidate out.mp4 --out runs/ep
rr-vfiqa evaluate --mode full-reference   --reference gt.mp4  --candidate out.mp4 --out runs/fr
```

### 同源模型比较

```bash
rr-vfiqa compare --mode endpoint-2x --reference src.mp4 \
  --candidates model_a.mp4 model_b.mp4 --out runs/compare
```

---

## 如何理解分数

| 字段 | 含义 |
|---|---|
| `overall_score` | 模式对应的工程质量或风险（0-100），**不是 MOS** |
| `confidence` | 本次评测条件和证据的可靠程度（0-1） |
| `status` | `ok` / `degraded` / `failed` |
| `scores` | 各维度子分数 |

```text
ok        → 核心流程完整
degraded  → 部分阶段或媒体导出退化，但仍有有效结果
failed    → 输入契约或核心指标不成立，overall_score = null
```

NR 的 `confidence` 上限为 0.75，因为它无法证明真实轨迹或显露区域内容。

---

## 三种评测模式

详细技术讲解见 [docs/index.html](docs/index.html)。

### No-Reference

- **输入：** 单条 60/120 FPS 视频
- **输出语义：** 时序稳定性和伪影风险
- **Schema：** `nr-stability-risk-v4`
- **能发现：** 冻结、交替模糊、抖动、UI 漂移、屏幕不稳定
- **不能证明：** 真实运动轨迹、显露背景、hallucination 是否正确

### Endpoint-Referenced

- **输入：** 60 FPS 端点参考 + 120 FPS 候选
- **输出语义：** 端点约束下的插帧质量
- **Schema：** `endpoint-reduced-reference-v3`
- **核心能力：** 双向 flow composition、reverse anchor cycle、MCT 残差、edge support、UI/文字/人物代理

### Full-Reference

- **输入：** 逐帧对应的 60 FPS GT + 60 FPS 候选
- **输出语义：** 同帧空间、时序和运动保真度
- **Schema：** `fr-same-rate-fidelity-v3`
- **核心能力：** Y/RGB L1、SSIM、边缘 F1/Chamfer、UI/文字 ROI 误差、光流差异、flicker excess
- **几何策略：** `strict`（默认，要求同分辨率）/ `resize-candidate` / `common-resolution`

---

## 速度、设备与可选后端

```bash
rr-vfiqa inspect --candidate out.mp4 --out runs/fast      --speed fast
rr-vfiqa inspect --candidate out.mp4 --out runs/balanced  -- speed balanced     # 默认，含 tier-3 审计
rr-vfiqa inspect --candidate out.mp4 --out runs/thorough  --speed thorough

rr-vfiqa inspect --candidate out.mp4 --device cuda --flow-backend raft    # GPU 精确光流
rr-vfiqa inspect --candidate out.mp4 --device cpu  --flow-backend farneback # CPU 默认
```

---

## 验证、性能与测试

```bash
python -m pytest                                          # CPU Farneback 路径
RR_VFIQA_TEST_FLOW=raft python -m pytest                   # GPU RAFT 路径
```

测试覆盖：数学方向单元测试、三模式路由、合成缺陷变形测试、媒体 E2E、Artifact 失败隔离、Farneback/RAFT 路径。

---

## 完整教程

需要完整的技术路线、三种模式原理、报告阅读指南、命令生成器和架构说明？

**[打开 docs/index.html](docs/index.html)**

---

## 项目状态和限制

- 公式融合未完成真实数据标定，`production_gate=false`
- NR 非 60/120 FPS 输入会 fail-closed
- FR 要求逐帧对应，同场景但非逐帧对应会拒绝打分
- NR/FR 不能判断视觉上合理的 hallucination
- 跨模式比较 `overall_score` 无意义
- 4K 和长视频完整性能矩阵尚未建立

---

## 常见问题

**Q: 我有参考视频，但帧率不是精确的 2× 怎么办？**
A: `inspect` 的 auto-safe 路由会检测 FPS 比例和对齐可靠性。不满足 2× 锚点对齐时会 fail-closed，不会猜测模式。

**Q: 两个模型的分数可以直接比较吗？**
A: 只有同一模式、同一参考、同一内容才可以比较。跨模式分数使用不同数学标尺，禁止直接排序。

**Q: 为什么 `overall_score` 是 null？**
A: 表示评测失败（`status=failed`）。查看 `report.html` 中的失败原因。常见原因：输入契约不成立、对齐不可靠、核心特征缺失。

**Q: 可以只评分不导出视频吗？**
A: 可以，使用 `--no-clips` 跳过坏例视频导出，速度更快。

---

## 开发与贡献

```text
rr_vfiqa/
├── io/                 视频读取与对齐
├── sampling/           全片扫描与窗口选择
├── motion/             独立光流后端
├── metrics/            模式核心指标
├── regions/            UI、文字、人物、细物体代理
├── fusion/             模式独立归一化与融合
├── diagnosis/          多证据规则与空间定位
├── report/             JSON、HTML、Heatmap、Clips
├── calibration/        标定和相关性验证
├── testing/            合成视频和参考插值器
└── execution/          Batch 等执行入口
```

最重要的使用原则：先确认手中的 reference 到底是"端点参考"还是"逐帧 Ground Truth"。`inspect` 的 auto-safe 路由会在证据不足时**拒绝猜测并拒绝打分**，因此可以放心地让程序自动判定。

---

当前版本：`0.4.0`
