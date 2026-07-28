# NoRefEval 完整验收结论

**结论：当前项目尚未达到我们之前预期的“可用于真实游戏插帧模型排序与质量门禁”的状态。**

它已经较完整地实现了技术路线的**架构性原型**：端点参考、风险采样、光流组合、运动补偿时序指标、奇偶帧分析、局部语义分支、融合评分和坏例报告均有对应模块。但当前 GitHub 提交存在阻断性文件缺失、多处指标实现错误、语义分支仍是弱代理、缺少真实数据标定和性能验证。

按当前远端仓库估算：

| 维度         |       达成度 | 判断                       |
| ---------- | --------: | ------------------------ |
| 技术架构与模块划分  |       80% | 基本符合原路线                  |
| 远端仓库可运行性   |       10% | 当前 clean checkout 无法正常导入 |
| 核心运动与时序指标  |       50% | 主体存在，但有坐标与方向错误           |
| 游戏特定坏例覆盖   |       30% | 多数是启发式代理，并非可靠语义检测        |
| 主观标定与排序可信度 |       10% | 仍是合成数据 bootstrap         |
| 评测效率       |       30% | 有级联思想，但解码和光流路径不高效        |
| 报告、CLI 与文档 |       70% | 基础体验较完整                  |
| **综合达成度**  | **约 40%** | **优秀原型，但未达到生产验收标准**      |

目前输出的 `overall_score` **不建议直接用于模型上线门禁、回归验收或模型能力定论**。

---

# 一、阻断级问题：当前 GitHub 仓库不能自洽运行

最严重的问题不是指标效果，而是源码包没有完整提交。

`pipeline.py` 和多个指标模块依赖：

```python
from .cache.source_cache import SourceCache
from ..cache.source_cache import SourcePairData
```

但我直接读取当前仓库中的：

```text
rr_vfiqa/cache/
rr_vfiqa/cache/source_cache.py
```

均得到 GitHub 404。

根因基本明确：`.gitignore` 中写了未锚定的：

```gitignore
cache/
```

这会同时忽略运行缓存目录和源码目录 `rr_vfiqa/cache/`。

而主流程在导入阶段就依赖该包。

这意味着很可能出现了：

> 本地工作区存在未跟踪的 `rr_vfiqa/cache/`，所以本地测试通过；提交 GitHub 时该源码目录被 `.gitignore` 整体过滤。

因此，当前远端提交无法证明 README 或提交信息中所称的测试结果。对应提交也没有可见的 CI 状态检查。

应立即改成：

```gitignore
/cache/
```

然后强制提交遗漏源码：

```bash
git add -f rr_vfiqa/cache
git commit -m "fix: include source cache package"
```

发布前必须在全新目录执行：

```bash
git clone ...
cd NoRefEval
pip install -e ".[dev]"
python -c "import rr_vfiqa"
pytest
```

---

# 二、已经正确落地的部分

项目并不是完全不可用的草稿。整体设计方向是正确的，而且代码组织与原技术路线高度一致。

主流程已经串联了：

* PTS 与锚点对齐；
* 全视频低成本扫描；
* 均匀采样与高风险采样；
* 源端特征缓存接口；
* 光流组合一致性；
* 反向锚点闭环；
* 运动补偿时序残差；
* 奇偶帧频率；
* 边缘结构；
* UI、人物、细物体、武器、文字和转场分支；
* 分类子分、总分、置信度和坏例报告。

这些模块在 `pipeline.py` 中已经形成完整调用链。

其中值得保留的设计包括：

1. **Endpoint-Referenced 定义正确。**没有错误地把任务当成纯 NR-VQA，而是充分使用原始 60 FPS 锚点。
2. **统一的 Schema 和模块边界较清晰。**后续替换光流、分割、跟踪和融合模型较方便。
3. **全片廉价扫描 + 风险窗口精评的方向正确。**
4. **运动补偿时序残差的主体实现基本合理。**中心帧到邻帧的 backward warp 及前后向循环可见性逻辑是当前实现中相对扎实的一部分。
5. **README 对标定状态相对诚实。**文档明确说明当前尺度和权重只是 synthetic bootstrap，仍需真实高帧率数据和人工 A/B 排序标定。

因此不需要推翻整个项目，主要需要修复数学实现、补齐语义能力并完成真实标定。

---

# 三、核心指标存在的正确性问题

## 1. 颜色变换方向写反

`estimate_color_transform(src, cand)` 拟合的是：

[
\text{cand}\approx gain\cdot \text{src}+offset
]

代码也确实以 source 为自变量、candidate 为目标拟合。

但锚点评测中却直接执行：

```python
c_norm = color.apply(c)
```

即再次对 candidate 应用 `gain * candidate + offset`，然后拿去与 source 比较。

正确的 candidate→source 归一化应该是：

[
\text{src}\approx
\frac{\text{cand}-offset}{gain}
]

这会影响颜色漂移残差、锚点完整性和对齐告警。

---

## 2. 多处半程投影混淆了 forward flow 与 backward warp

项目定义的：

```python
warp_image(img, flow)
```

语义是：

[
out(x)=img(x+flow(x))
]

也就是 backward sampling。

但反向锚点闭环使用：

```python
wa = warp_image(img_a, 0.5 * f_ab)
```

其中 `f_ab` 是从 A 指向 B 的 forward flow。对于一个向右平移的物体，这会把 A 向错误方向移动。

相同模式也存在于：

* endpoint edge support；
* 人物预期 mask 投影；
* 人物 endpoint leak 计算。

例如边缘支持直接执行：

```python
s0 = warp_image(e0, 0.5 * f_01)
```

同样把 forward flow 当成了目标网格 backward flow。

人物分支中也使用相同方式生成中间 mask，并使用定义在 anchor 网格上的 flow 去对齐 generated-frame 网格。

建议彻底禁止模糊的 `warp_image(img, flow)` 调用方式，明确拆成：

```python
backward_warp(source, target_to_source_flow)
forward_splat(source, source_to_target_flow)
```

并对恒定平移建立数学单元测试。

---

## 3. 双向光流组合使用了错误坐标系的遮挡权重

forward composition 位于 (X_i) 网格，backward composition 位于 (X_{i+1}) 网格。

但当前实现给 forward 和 backward 两个方向使用同一张 `conf_01/occ_01` 权重图。

backward 分支应使用：

```python
conf_ba
occ_ba
```

否则遮挡边界和显露区域的权重会被映射到错误空间，尤其会影响大幅旋转和人物遮挡场景。

---

## 4. UI 边缘 F-score 基本退化为恒定 1

当前实现中：

```python
da = distance_to_edges_a
db = distance_to_edges_b

prec = db[edges_b]
rec  = da[edges_a]
```

而边缘 B 上到边缘 B 自身的距离必然为零，边缘 A 上到边缘 A 自身的距离也必然为零。因此，只要双方都有边缘，precision 和 recall 基本都会变成 1。

正确写法应是：

```python
precision = mean(distance_to_a[edges_b] <= tol)
recall    = mean(distance_to_b[edges_a] <= tol)
```

这个错误同时影响：

* UI 边缘完整性；
* 文字边缘完整性；
* UI/text 子分；
* 坏例分类。

---

## 5. UI 和转场的“双重曝光”条件数学上几乎不可能触发

UI 动态分支要求：

```python
d0 < 8
d1 < 8
diff01 > 24
```

但根据三角不等式：

[
d(X_i,X_{i+1})
\leq d(X_i,M)+d(M,X_{i+1})<16
]

因此不可能同时大于 24。

转场分支同样要求：

```python
d0 < 10
d1 < 10
diff01 > 24
```

也基本不可能在同一 changed pixel 上成立。

真实 Alpha 混合帧的特征不是“同时非常接近两个端点”，而是：

[
M\approx \alpha X_i+(1-\alpha)X_{i+1}
]

应通过最优 (\alpha) 拟合、双边缘、局部梯度衰减和残差结构检测。

---

## 6. UI regression 的方向相反

当前动态 UI 逻辑在中间帧逐渐远离初始状态 (X_i) 时增加惩罚：

```python
sim_cur_to_xi - sim_prev_to_xi
```

但对于正常的前向状态变化，当前生成帧本来就应比之前的生成帧更远离 (X_i)。这段代码实际上会把正常推进识别为回退。

---

## 7. 武器跟踪从错误帧初始化

武器分支在 `grays[1]` 上提取关键点：

```python
pts = _corner_points(grays[1], ...)
```

随后把这些点传给 KLT。

但 KLT 实现把传入点当作 `frames_gray[0]` 上的初始点，并直接从 frame 0 跟踪到 frame 1。

这使整条轨迹从第一步开始就处于错误坐标系。

此外，pipeline 没有把人物 ROI mask 传给武器分支，所以它实际跟踪的是画面中央的普通角点，而非武器点。当前 `weapon_dev_p90` 更准确的名称应是“中央区域局部点轨迹偏差”，不能可靠解释成剑尖或剑柄抖动。

---

## 8. LightGBM 校准器存在 100 倍量纲错误

测试使用的是 0～100 的主观分数：

```python
y = 100 * exp(...)
cal.fit(X, y)
```

但推理时又将模型输出乘以 100：

```python
raw = model.predict(...)
return clip(raw * 100, 0, 100)
```

因此如果模型学到的是 60、80、90 这样的 MOS，最终全部会被裁剪到 100。当前 monotonic test 可能因为 `base` 和 `worse` 都饱和为 100 而表面通过。

此外，`fit_pairwise()` 并没有实现真正的 pairwise ranking loss，只是把 winner 标为 1、loser 标为 0 后做普通回归。

---

# 四、语义坏例覆盖情况

## 人物身体缺失：只有弱代理

目前没有真正的人体、部位或游戏角色分割模型。默认分割器是“残余运动 + 局部对比度”的 classical proxy。代码本身也明确说明训练好的游戏域模型尚不存在。

因此它不能稳定区分：

* 人物和其他移动前景；
* 头部、手臂、腿部；
* 武器和角色本体；
* 静止人物和静止背景；
* 同相机运动的人物。

当前只能认为“人物完整性接口已建立”，不能认为“人物缺头、缺腿检测已经实现”。

## 柱子和细物体：有启发式能力，但泛化不足

薄物体模块使用 LSD 长线段和线段内外光流差，方向上符合之前提出的运动层归属误差。

但当前问题包括：

* 所有长直线都会进入，包括建筑纹理和道路边缘；
* 只从前端点检测，只检查前半程；
* 没有双向遮挡处理；
* 没有实例时序关联；
* line count 是全画面统计，不是同一个柱子或武器的持续跟踪；
* 输出的 `box` 实际是线段端点，未必满足标准 bbox 格式。

所以这一分支适合风险触发，不足以作为可靠的细物体质量分。

## UI 和文字：框架存在，但关键计算当前不可用

UI mask 使用跨视频静态像素和持久边缘构建，适合固定 HUD，但无法可靠覆盖：

* 技能冷却动画；
* 临时商店与弹窗；
* 持续变化的数字；
* 场景切换后的不同 UI；
* 只在局部片段出现的文字。

文字分支依赖已经出错的 `_edge_fscore`；而所谓 ROI 内 Otsu 实际是在整张灰度图上进行阈值分割，并非 ROI 内阈值。

## 卡牌和商店转场：尚未达到预期

当前没有：

* 卡牌四角或 homography；
* 翻牌角度单调性；
* 商店组件级状态跟踪；
* 可靠 Alpha 混合检测；
* 前进/回退状态序列建模。

因此只能检测一部分大面积像素变化，不能宣称已完成棋牌和商店状态劣化评测。

---

# 五、三级级联与评测效率并未真正实现

配置中定义了：

* `audit_top_fraction`；
* `audit_max_windows`；
* `full_res_edges`；
* `run_tracker`；
* `run_depth`。

但 pipeline 没有真正执行“二级筛选后再升级少量窗口”的 Audit 流程。只要开启 region branches，就会对所有选中窗口运行人物、薄物体、文字、转场和 weapon tracker。

这意味着：

* `standard` 的 `run_tracker=False` 没有生效；
* `audit_top_fraction` 没有生效；
* `audit_max_windows` 没有生效；
* `full_res_edges` 没有生效；
* 深度分支没有接入；
* CoTracker 仍然只是抛出 `NotImplementedError` 的占位接口。
* VQA 后端默认直接返回 `None`，FAST-VQA、DOVER、VFIPQA 均未接入。
* 深度后端同样只是接口占位。

## 解码路径会成为严重瓶颈

`read_frames()` 每次随机读取都从视频开头解码，直到所需的最后一帧才停止。

项目中以下操作都会反复调用它：

* 锚点校验；
* UI 14 帧采样；
* 每个评测窗口读取；
* compare 多候选重复初始化。

长视频靠后的窗口会重复解码前面的大量帧，实际复杂度接近：

[
O(N_{\text{windows}}\cdot N_{\text{video frames}})
]

而不是预期的一次顺序解码。

RAFT 也没有批处理；每个 flow pair 会分别执行正向和反向两次模型推理。

标准模式最多约 48 个窗口，每个窗口需要多个 pair，单卡上很可能远慢于方案目标。目前仓库中没有 1080p、4K、长视频、单卡显存或多候选缓存收益的实测报告。

所谓 `test_cache_speeds_second_run` 也没有测量速度，只检查两次分数差小于 2。

---

# 六、测试和“充分评测”结论不成立

当前端到端测试全部使用 `fast` 预设。

而 `fast` 明确关闭全部 region branches。

因此端到端测试没有覆盖：

* 人物分支；
* 细小物体分支；
* 武器跟踪；
* UI；
* 文字；
* 卡牌或商店转场；
* Audit 流程。

合成坏例也只有三类：

* 高斯模糊；
* 前后帧 crossfade ghost；
* 直接复制前帧 freeze。

没有覆盖最初提出的关键坏例：

* 大幅旋转背景撕裂；
* 背景显露区域丢失；
* 头部或腿部缺失；
* 柱子跟随背景；
* 剑尖局部闪烁；
* UI 亚像素漂移；
* 文字笔画粘连；
* 卡牌翻转；
* 商店状态跳变；
* 场景切换；
* VFR、丢帧和时间戳偏移。

项目 README 自己也明确承认：

* 当前尺度只是 synthetic bootstrap；
* 还没有真实 120/240 FPS 伪 GT 标定；
* 还没有真实 60→120 人工 A/B 排序标定。

所以当前测试只能证明：

> 在一个约 1.3 秒、320×192、单一合成场景上，fast 模式可以把混合了 blur/ghost/freeze 的候选排在 perfect interleave 之后。

它不能证明评测器能够可靠评价真实游戏插帧。

---

# 七、评分与置信度还有“失败反而高分”的风险

pipeline 对每个指标阶段使用：

```python
try:
    wf.scalars.update(fn())
except Exception:
    wf.labels["error_xxx"] = ...
```

失败阶段产生的错误不会进入最终报告，也不会直接降低分数。由于 fusion 只聚合已有 feature，某个困难分支失败后，它的坏分数反而消失。

更严重的是，如果没有任何窗口成功：

* 各类别误差会变成 NaN；
* 总分函数为缺失类别使用默认误差 0.15；
* 总分仍约为 86；
* confidence 对空窗口也没有加入明确的“零窗口”惩罚。

默认缺失类别参与总分的逻辑见：
置信度逻辑见：

这会导致严重的 fail-open：

> 评测没有真正完成，却仍可能得到较高总分和较高置信度。

生产评测必须改成 fail-closed：核心指标缺失时不输出有效总分，或者将 confidence 降到接近零。

---

# 八、建议的整改顺序

## P0：恢复远端仓库可运行性

必须先完成：

1. 将 `.gitignore` 的 `cache/` 改为 `/cache/`；
2. 提交 `rr_vfiqa/cache/` 全部源码；
3. 添加 clean-checkout CI；
4. CI 中执行安装、import、CPU 测试和 GPU 可选测试；
5. 将指标阶段异常写入报告；
6. 核心阶段失败时禁止输出正常总分。

在这一步完成前，不应继续调整指标权重。

## P1：修复数学和坐标实现

按优先级修复：

1. 统一 forward splat 与 backward warp 语义；
2. 修复 cycle、edge、character 的半程投影；
3. 修复双向 composition 的 backward 遮挡权重；
4. 修复颜色变换方向；
5. 修复 UI edge F-score；
6. 重写 UI/transition 的 Alpha 混合检测；
7. 修复 UI regression 符号；
8. 修复 KLT 初始化帧；
9. 修复 calibrator 的 0～1 / 0～100 量纲；
10. 将 `char_leak_mean` 的归一化方向重新定义。

每一项都应增加确定性单元测试，而不是只测试分数单调性。

## P2：真正实现语义能力

需要至少接入：

* 游戏域人物/角色分割模型；
* 人物部位或关键区域检测；
* character mask 向 weapon tracker 传递；
* 细物体双向实例跟踪；
* UI 组件级模板或检测器；
* 卡牌四边形和 homography；
* CoTracker 或等价的审核级跟踪器；
* 高视差窗口的深度分层。

启发式分支可以保留为 Fast 模式，但不能用语义名称包装弱代理结果。

## P3：完成真实标定和性能验收

至少需要三套数据：

1. **真实 120/240 FPS 伪 GT：**用于验证端点指标与真实中间帧误差的关系；
2. **真实模型输出：**覆盖不同游戏、不同运动和多个插帧模型；
3. **人工 A/B 排序：**用于最终融合和阈值标定。

最终报告应包含：

* SRCC；
* PLCC；
* Pairwise Accuracy；
* 各坏例类别 AP/F1；
* 最差 10% 坏例召回率；
* Leave-One-Game-Out；
* Leave-One-Model-Out；
* 60 秒 1080p/4K 单卡耗时；
* 峰值显存；
* 第 2、3、N 个候选相对于首候选的缓存加速比。

---

# 最终判断

**NoRefEval 已经完成了一个较好的技术方案代码化原型，但没有完成一个经过充分验证的无中间帧参考插帧质量指标。**

当前最适合的定位是：

> **Research prototype / metric development framework**

而不是：

> **Validated VFI quality metric / production evaluation gate**

核心架构可以保留，尤其是风险采样、运动补偿、光流组合和可解释报告设计；但在修复仓库完整性、坐标数学错误、UI/转场逻辑、武器跟踪、校准器和真实数据验证之前，现有总分还不具备可信的模型排序意义。
