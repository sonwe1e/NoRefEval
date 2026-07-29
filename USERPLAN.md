# 总体结论

我审查的是 `codex/norefeval-p0-reliability` 当前最新提交：

```text
03c3c3b8a5f25b1bad50900b1369da9a5acbf896
feat: implement USERPLAN reliability contracts
```

该分支目前比 `main` 多 3 个提交。最新版本已经落实了上一轮路线中的大部分工程可靠性要求，尤其是 NR 时间尺度、FR 全片参考扫描、Feature Registry、显式运行契约和 Endpoint 类别有效窗口等关键工作。

当前项目可以定位为：

> **一个架构较完整、数学约束明显改善、可以正式进入真实数据验证阶段的多模式 VFI 评测研究平台。**

但还不能定位为：

> **已经验证准确、可以直接作为模型上线门禁的质量评测产品。**

综合完成度判断：

| 维度                  |    上一版本 |        当前版本 |
| ------------------- | ------: | ----------: |
| 多模式架构               |     90% |     **95%** |
| 可复现契约               |     55% |     **80%** |
| No-Reference 算法实现   | 55%～60% | **70%～75%** |
| Endpoint-2x 算法实现    |     80% |     **85%** |
| Full-Reference 算法实现 | 65%～70% | **78%～82%** |
| 自动测试覆盖              |     55% |     **70%** |
| 独立模式验证体系            |     15% |     **35%** |
| 性能与资源验证             |     40% |     **45%** |
| 真实数据标定              | 15%～20% |  **仍约 20%** |
| 生产门禁能力              | 35%～40% |   **约 45%** |

---

# 一、上一轮计划的实际完成情况

## 已完成

当前版本已经完成了以下关键任务。

| 上一轮任务                         | 当前状态    |
| ----------------------------- | ------- |
| 强制显式指定评测模式                    | 已完成     |
| NR 默认关闭环境相关的 `auto` VQA       | 已完成     |
| 可选 NIQE 使用独立 Schema 后缀        | 已完成     |
| unique-frame coverage         | 已完成     |
| Feature Registry              | 已完成基础版本 |
| MetricResult 契约               | 已建立基础结构 |
| NR TemporalLagPlan            | 已完成     |
| NR FlowPairPlan 批量光流规划        | 已完成     |
| NR 使用 native、1/60、1/30 三档 lag | 已完成     |
| NR 统一物理时间 self-reference      | 已完成     |
| NR acceleration/jerk 使用真实时间   | 已完成     |
| NR composition/cycle 遮挡加权     | 已完成     |
| NR 原生 120 FPS 重复帧检测           | 已完成     |
| NR compare 内容一致性检查            | 已完成     |
| FR 显式 geometry policy         | 已完成     |
| FR 全片低分辨率参考扫描                 | 已完成     |
| FR 标准局部 SSIM                  | 已完成     |
| 伪 perceptual 指标重命名            | 已完成     |
| Endpoint 按类别排除失败阶段窗口          | 已完成     |
| NR/FR 独立方向性验证入口               | 已完成基础版本 |
| 三模式方向性 CI 测试                  | 已加入     |

最新提交还为报告增加了：

* `metric_contract`；
* `preset_contract`；
* `feature_contract_hash`；
* `backend_contract`；
* `code_commit`；
* `working_tree_dirty`；
* `production_gate=false`。

这些信息显著提高了实验可复现性。

## 部分完成

以下任务已有实现，但还没有完整闭环：

* MetricResult 目前主要用于检查 required scalars，maps、instances、coverage 和 confidence 尚未真正参与执行。
* Feature Registry 已建立，但单位和分辨率属性仍有启发式推断。
* NR 区域诊断增加了光流几何和 KLT，但还不是完整的人物、细物体和武器评测。
* FR 全片扫描已经进入采样和融合，但实现方式存在明显内存与重复解码问题。
* NR/FR 验证 harness 已存在，但没有提交真实方向性结果。
* CI 增加了方向性测试，但 GitHub 当前没有返回该提交的 workflow run。

---

# 二、No-Reference 模式评估

## 本轮取得的实质进步

### 1. 60/120 FPS 的公共时间尺度已经修正

新建的 `TemporalLagPlan` 会按 PTS 构造：

```text
native cadence
1/60 秒
1/30 秒
```

对于 120 FPS：

```text
1/60 秒：跨 2 帧
1/30 秒：跨 4 帧
```

对于 60 FPS：

```text
1/60 秒：跨 1 帧
1/30 秒：跨 2 帧
```

公共 self-reference triplet 也固定为中心前后各 1/60 秒，总跨度 1/30 秒。这解决了上一版 60 FPS 与 120 FPS 使用不同物理跨度，却共用同一个特征名的问题。

### 2. Flow 推理规划得到改善

`FlowPairPlan` 会收集：

* native triplet；
* common triplet；
* native MCT；
* 1/60 MCT；
* 1/30 MCT；

所需的全部 frame pair，再交给 `WindowFlows.precompute()` 批量计算双向 flow。这样避免了指标执行过程中反复触发零散 RAFT 调用。

### 3. Self-composition 和 self-cycle 已加入遮挡处理

当前会根据外侧帧双向 flow 计算：

* forward/backward confidence；
* occlusion；
* endpoint-grid visibility weight；
* splat 后的可见性。

这比上一版只依靠 splat coverage 更合理，快速运动和显露区域不再与正常可见区域完全混在一起。

### 4. 运动异常诊断明显增强

新增了：

* tile-wise velocity；
* acceleration；
* jerk；
* local reversal；
* flow folding；
* Jacobian determinant；
* divergence；
* curl；
* KLT camera-relative track acceleration；
* track jerk；
* track direction change。

这些信号使 NR 不再只依赖全图清晰度和运动补偿残差，对局部运动异常的感知能力有所提高。

### 5. 120 FPS 单帧复制问题已有直接测试

当前测试构造：

```text
Y1 = Y0
Y3 = Y2
...
```

并要求：

* `nr_native_duplicate_fraction` 明显升高；
* duplicated 视频的 `temporal_stability` 低于 clean。

这是 NR 模式第一次拥有明确的坏例方向性测试，而不只是验证“可以运行”。

---

## NR 仍存在的核心问题

### 1. 60 与 120 FPS 仍不真正共享同一总分尺度

虽然公共时间尺度已经加入，但评分 Schema 仍同时使用：

* `nr_common_*`；
* `nr_native_*`。

native 特征在：

| FPS     | native 时间跨度 |
| ------- | ----------: |
| 60 FPS  |    16.67 ms |
| 120 FPS |     8.33 ms |

因此 `nr_native_self_cycle`、`nr_mct_native_mean`、native duplicate 等特征在两种 FPS 下仍不属于同一统计分布，但目前二者使用相同的：

```text
nr-stability-risk-v2
```

这是当前最重要的 NR 分数语义问题。

推荐二选一：

1. **公共分数只使用 1/60 和 1/30 特征，native 特征仅作为诊断项；**
2. 分成：

   ```text
   nr-stability-risk-v2-60
   nr-stability-risk-v2-120
   ```

当前 NR compare 会禁止 60 和 120 混合排序，所以相对排序风险已经降低，但单份报告中的 60 分和 120 分仍不宜直接比较。

### 2. “Camera-relative”目前只移除了平移

KLT 跟踪当前每一帧只减去全部点的 median position，本质上只消除了全局平移。

对于游戏中的：

* 相机旋转；
* 缩放；
* 透视变化；
* 大幅摇杆转向；

正常轨迹仍会在相对坐标中产生加速度和方向变化，可能被误判为局部抖动。

更合理的做法是先估计：

```text
translation → affine → homography
```

再把 track 变换到 camera-stabilized 坐标。

### 3. NR UI 仍然不符合手游 UI 的主要布局

当前 UI 区域只取：

* 顶部 1/5；
* 底部 1/5。

但你的主要游戏场景中常见的是：

* 左下移动轮盘；
* 右下技能按钮；
* 左右两侧 HUD；
* 屏幕中部悬浮文字。

因此现在的 NR UI/text 指标会漏掉大量关键 UI。

应改为：

* 四边 border band；
* 全屏 screen-static mask；
* persistent edges；
* 排除大面积静态场景；
* connected-component 级 UI 分析。

### 4. NR compare 可能过度拒绝真正需要比较的候选

当前 compare 会使用：

* pHash；
* 低分辨率灰度 L1；
* FPS；
* 时长；
* PTS；
* 分辨率；

判断候选是否同内容。

其中灰度缩略图 L1 阈值约为 12。严重模糊、亮度漂移或色彩错误的模型恰恰可能超过这个阈值，从而被认为“不是同一个内容”，无法排序。

也就是说：

> 候选质量越差，越可能被比较保护机制拒绝。

建议内容一致性只使用对质量变化更鲁棒的证据：

* histogram-normalized pHash；
* DINO/ConvNeXt 低频语义特征；
* camera trajectory fingerprint；
* 用户显式提供 `comparison_group_id`。

### 5. NR 仍没有真实有效性证据

目前只有：

* 60/120 能运行；
* 120 FPS duplicate 方向正确；
* 特征存在性；
* 输入契约。

尚未验证：

* blur severity 单调性；
* ghost；
* rotation tear；
* UI drift；
* text instability；
* weapon flicker；
* camera rotation false positive；
* 真实模型 A/B 排序。

因此 NR 当前完成度应理解为：

> **算法框架约 75%，指标可信度约 30%。**

---

# 三、Endpoint-2x 模式评估

Endpoint 仍然是项目最成熟的模式。

## 本轮主要改进

新增了 `CATEGORY_REQUIRED_STAGES`。例如：

```text
motion      依赖 composition
temporal    依赖 cycle + temporal + parity
structure   依赖 edges
character   依赖 character
ui          依赖 ui + text
```

如果某窗口的 temporal 阶段异常，该窗口的残余 temporal 特征不会再参与 temporal 类别聚合。

同时报告增加了：

```text
category_valid_windows
```

便于识别每个类别到底有多少有效窗口。

## 当前仍有三个主要问题

### 1. Score Schema 应该升级到 V2

Endpoint 的类别聚合逻辑已经发生改变：

* 旧版会使用阶段失败窗口的残余特征；
* 新版会按类别排除失败窗口。

这意味着同一输入在旧版和新版可能得到不同总分。

但报告仍然写：

```text
endpoint-reduced-reference-v1
```

而只把 metric contract 改成：

```text
endpoint-metrics-v2
```

为了保持严格可复现，应把 Score Schema 同步升级为：

```text
endpoint-reduced-reference-v2
```

### 2. 类别有效性只检查异常，不检查特征完整性

当前逻辑主要检查：

```text
error_<stage>
```

但如果一个阶段正常返回，只是大量关键特征为 NaN，类别仍可能使用剩余少数特征。

Endpoint 也应采用 MetricResult 或类别 required-feature contract，例如：

```text
motion 至少要求 comp_mean
temporal 至少要求 mct + cycle 中各一个有效量
structure 至少要求 edge_recall/precision
```

### 3. 误报问题没有根本变化

最新提交主要是可靠性和架构改造，没有重新训练阈值或替换语义代理。

当前仓库记录的严格 12 类合成定位结果仍为：

* Recall：0.667；
* Precision：0.229；
* F1：0.340。

所以 Endpoint 目前依然适合：

* 同源模型探索性排序；
* 找可疑片段；
* 辅助人工分析。

还不适合把具体标签直接当成确定语义结论。

---

# 四、Full-Reference 模式评估

FR 是本轮提升最明显的模式。

## 已完成的重要改进

### 1. 加入全时间线 Reference Scan

当前会在低分辨率全片计算：

* Y L1；
* chroma L1；
* gradient/Laplacian 差异；
* edge mismatch；
* SSIM proxy；
* frame-difference mismatch。

该风险会参与：

* 风险窗口选择；
* 全局融合。

这解决了之前只根据 candidate 自身风险抽样、可能漏掉“稳定但始终错误”问题的缺陷。

### 2. 空间指标更符合 Full-Reference 定义

新增或修正了：

* Y L1；
* RGB L1；
* RGB Charbonnier；
* PSNR；
* 11×11 Gaussian local SSIM；
* edge recall；
* edge precision；
* edge F1；
* edge Chamfer；
* multiscale luma + gradient L1。

原来的 `fr_multiscale_perceptual` 已移除，不再把普通多尺度 L1 描述成感知指标。

### 3. 新增局部 ROI 参考误差

现在会构造：

* UI proxy ROI；
* text proxy ROI；
* salient structure ROI；
* center motion ROI。

并计算对应局部误差。尽管它们仍是 proxy，但比全图平均更不容易淹没小目标问题。

### 4. Geometry policy 已显式化

当前支持：

```text
strict
resize-candidate
common-resolution
```

不同策略会使用不同 score schema 后缀，并保存 resize transform。这个设计是正确的。

---

## FR 当前仍存在的关键问题

### 1. 全片扫描存在明显内存问题

`scan_full_reference()` 当前直接执行：

```python
reference.decode_all(width=width)
candidate.decode_all(width=width)
```

也就是把两条完整视频全部堆叠在内存中。

即使缩放至 320 宽，60 秒、60 FPS 的两条视频也可能消耗超过 1 GB 内存；长视频会线性增长。

而且 FR 目前会进行多次完整解码：

1. candidate cheap scan；
2. reference alignment descriptor；
3. candidate alignment descriptor；
4. reference full scan；
5. candidate full scan；
6. 后续窗口随机读取。

这会成为 FR 的主要性能瓶颈。

必须改为：

> 单次流式顺序解码，同时生成 descriptor、cheap feature 和 FR scan。

### 2. `common-resolution` 仍有尺寸不一致风险

该策略目前只选择共同宽度，然后分别让 VideoReader 按原始宽高比缩放。

如果两条视频的宽高比在允许误差内但并非完全相同，例如：

```text
1920×1080
1280×718
```

二者缩放到相同宽度后高度可能不同，最终仍会在 `compute_window()` 中触发 shape mismatch。

应显式计算共同目标：

```text
target_width
target_height
```

然后两路都强制 resize 到完全相同的网格。

### 3. `fr_scan_gradient_l1` 名称与实现不一致

全片扫描里实际计算的是：

```python
cv2.Laplacian(...)
```

而特征名和文档使用 `gradient_l1`。

应二选一：

* 改为 Sobel gradient magnitude；
* 或更名为 `fr_scan_laplacian_l1`。

这关系到 Feature Contract 的准确性。

### 4. FR 参考侧没有跨候选缓存

当比较多个 candidate 时，每个候选都会重新计算：

* reference descriptor；
* reference full scan；
* reference window flow；
* reference ROI。

尚未实现计划中的：

[
T_\text{reference}+N\cdot T_\text{candidate}
]

FR 多模型比较的效率仍有较大优化空间。

### 5. Motion fidelity 仍以全图 median flow 为主

`fr_trajectory_deviation` 仍主要累积全图 median flow，容易被 camera motion 主导。

更合理的是：

* 4×4/8×8 tile flow；
* camera residual flow；
* foreground/salient ROI flow；
* track trajectory；
* P50/P90/P99 局部误差。

---

# 五、公共工程质量评估

## Feature Registry

这是正确的长期方向，目前已经记录：

* name；
* mode；
* category；
* units；
* direction；
* resolution invariant；
* required；
* scale；
* version。

并可生成 feature contract hash。

不过当前单位是根据名字推断的。例如 `fr_l1_rgb` 可能被推断成 luma 单位。长期应改为每个特征显式注册，而不是字符串猜测。

## MetricResult

目前已经有统一结构：

```python
MetricResult(
    scalars,
    maps,
    instances,
    coverage,
    confidence,
    warnings,
    status,
)
```

但当前实际只使用：

* scalars；
* warnings；
* status。

以下字段尚未形成真实闭环：

* maps；
* instances；
* metric coverage；
* metric confidence。

而且 metric warnings 只存放在窗口 labels 中，没有汇总到最终报告，发生 required feature 缺失时使用者不容易看到具体原因。

## Provenance

当前记录已经明显改善，但仍有两个问题：

1. 从 wheel 或非 Git checkout 部署时，运行目录可能没有 `.git`，此时 `code_commit` 会变成 `unknown`。
2. Endpoint 外部 LightGBM 校准器只记录 `lightgbm-external`，没有记录文件 SHA256。

建议在构建包时嵌入 commit，并记录：

```text
calibrator_path
calibrator_sha256
calibrator_feature_contract_hash
training_manifest_hash
```

---

# 六、测试与验证评估

## 已有进步

CI 现在额外强调三模式方向性：

* Endpoint bad < good；
* NR duplicate < clean；
* FR bad < good。

测试还覆盖：

* TemporalLagPlan 物理时间；
* FlowPairPlan；
* unique coverage；
* Endpoint 类别失败窗口过滤；
* local SSIM identity；
* NR compare 拒绝不同内容；
* FR 三种 geometry policy；
* Feature Contract；
* MetricResult required fields。

## 尚不充分

当前 mode validation harness 只计算：

```text
better_score > worse_score
```

它没有检查：

* confidence；
* `meta.status`；
* 最差窗口位置；
* 缺陷标签；
* 分数差距；
* 统计显著性；
* 每特征方向性；
* 跨游戏泛化。

同时，目前没有提交实际 NR/FR manifest 运行结果。新增的 `tests/test_mode_validation.py` 使用的是 mock report，只证明 harness 代码能够工作，不证明指标能够工作。

此外，本次我尝试在隔离环境执行全新 clone，但环境无法解析 `github.com`，因此无法独立运行 pytest。GitHub 也没有返回最新提交的 workflow run。测试结论目前来自代码定义，而不是本次独立复现。

---

# 七、性能评估

目前已有性能数据仍主要是旧的 Endpoint Fast/RAFT：

* 60 秒 1080p；
* 约 65 秒；
* 峰值显存约 907 MB。

新版本新增的 NR/FR 路径尚无实测。

预计主要风险：

| 模式         | 主要开销                             |
| ---------- | -------------------------------- |
| NR-120     | 每窗口约 40 个有向 flow、KLT、geometry    |
| FR         | 多次全片解码、双路 flow、全片内存堆叠            |
| FR compare | reference 侧重复计算                  |
| Audit      | native-resolution RAFT 与 tracker |

因此目前不能认为 Standard/Audit、NR-120 或 FR 已满足长视频生产效率要求。

---

# 八、当前最终验收矩阵

| 目标               | 当前判断          |
| ---------------- | ------------- |
| 三种模式显式分离         | **完成**        |
| NR 60/120 固定时间尺度 | **基本完成**      |
| NR 120 单帧复制检测    | **完成基础能力**    |
| NR 遮挡感知          | **完成**        |
| NR 局部运动几何        | **完成基础能力**    |
| NR 游戏人物/武器/侧边 UI | **未充分完成**     |
| NR 真实模型排序        | **未验证**       |
| Endpoint 工程可靠性   | **基本完成**      |
| Endpoint 类别级失败隔离 | **完成基础能力**    |
| Endpoint 语义标签精度  | **仍不足**       |
| FR 全片参考风险扫描      | **完成，但实现需优化** |
| FR 标准空间指标        | **基本完成**      |
| FR 局部 ROI 评测     | **完成代理版本**    |
| FR 多候选高效共享       | **未完成**       |
| 三模式可复现契约         | **基本完成**      |
| 三模式真实校准          | **未完成**       |
| 自动生产上线门禁         | **不满足**       |

---

# 九、下一步优先级

下一版本应停止继续大规模新增指标，优先解决以下事项。

## P0：必须先修

1. NR 公共总分排除 native 特征，或者拆分 60/120 Schema。
2. Endpoint Score Schema 升级到 V2。
3. 修复 FR `common-resolution` 的显式目标宽高。
4. 将 FR full scan 改成流式处理，禁止 `decode_all()` 堆叠全视频。
5. 将 `fr_scan_gradient_l1` 实现或名称修正一致。
6. 汇总 MetricResult 失败原因到报告。
7. 给校准器和构建版本增加 SHA256。

## P1：决定指标有效性

1. 建立 NR blur/freeze/ghost/tear/UI/weapon 方向性数据。
2. 建立 FR blur/color/shift/delete/freeze/codec 方向性数据。
3. 测试正常 camera rotation 对 NR 的误报。
4. 将 NR UI mask 扩展到左右两侧和全屏静态组件。
5. 改善 NR compare 的内容指纹，避免拒绝严重坏模型。
6. 建立 FR reference cache。

## P2：真实验收

分别提交：

```text
validation/nr_real.json
validation/endpoint_real.json
validation/fr_real.json
```

至少报告：

* SRCC；
* PLCC；
* pairwise accuracy；
* per-defect precision/recall/F1；
* worst-10% recall；
* 60/120 分布；
* runtime；
* peak VRAM；
* peak RAM。

---

# 最终判断

**这一版本已经实质性完成了上一轮提出的可靠性路线，项目无需再次推翻架构。**

当前最准确的阶段定位是：

> **多模式算法开发基本完成，正式进入“证明这些指标是否有效”的阶段。**

从代码成熟度看，项目已经可以开始接入真实游戏数据进行系统实验；但在以下三件事完成前，仍不能启用生产门禁：

1. NR 60/120 分数尺度问题得到解决；
2. NR/FR 真实模型方向性结果达到稳定门槛；
3. FR 流式性能和 Endpoint 误报率得到明显改善。
