核心判断

这两次 no-reference 结果的 Overall Quality 8.8 和 8.0 当前都不具备有效解释性。它们几乎完全是被错误偏高的 Cadence 风险压低的：

69.1 \times 12.8\% \approx 8.8

64.9 \times 12.3\% \approx 8.0

因此，这不是“原始视频只有 8.8 分、插帧视频只有 8.0 分”，而是：

当前 60 FPS Cadence 判定把正常战斗运动、特效和相位变化误识别成了接近完全的帧率塌缩。

短期内应忽略这两项结果中的：

* Overall Quality
* Cadence Integrity
* cadence risk
* Quality Level
* Affected

更有参考价值的是 Common-time 和各子分数，但其中 phase_consistency 也存在明显的 60 FPS 适用性问题。

⸻

一、这组结果实际说明了什么

先排除 Cadence 惩罚，原始视频与插帧视频的基础分数是：

指标	原始 60 FPS	插帧 60 FPS	变化
Common-time	69.1	64.9	-4.2
Temporal stability	88.4	87.5	基本相同
Motion smoothness	37.3	38.5	插帧略高
Phase consistency	42.7	24.5	插帧明显更低
UI/text stability	27.7	27.9	基本相同
Technical quality	56.3	49.0	插帧略差

当前 NR 融合权重为：

temporal 35%
motion   30%
phase    20%
UI       10%
technical 5%

子分数和总分又都通过指数函数融合。

按照当前公式从你提供的子分数逆算：

如果暂时去掉 phase category，两个视频的基础分数约为 70.86 和 70.84，几乎完全相同。

也就是说，原始与插帧视频 4.2 分的 Common-time 差距，几乎全部来自：

phase_consistency：42.7 → 24.5

而不是运动平滑度或时序稳定性。

对当前视频的合理解释

从这组数据能够相对可信地得出：

* 插帧没有明显降低普通的时间连续性，temporal_stability 基本不变。
* 插帧没有让当前光流运动指标明显恶化，motion_smoothness 反而略高。
* 插帧帧与原始帧之间存在明显的奇偶相位差异，可能表现为生成帧更模糊、边缘密度不同、锐度不同或编码特征不同。
* 插帧视频的绝对技术质量略差，可能有锐度损失、压缩损失或噪声差异。
* UI 分数对两个视频都异常低，说明它主要反映当前检测器对战斗特效和动态 HUD 的误判，而不是插帧引入的差异。

因此，当前唯一值得重点检查的真实差异是相位一致性和技术质量，而不是 Cadence 或 Motion。

⸻

二、为什么两个 60 FPS 视频都会得到 0.93～0.95 Cadence Risk

这是当前实现中的结构性问题。

1. 60 FPS 下 native 和 1/60 实际是同一个时间尺度

当前时间规划同时构造：

native：相邻帧
lag_1_60：相隔 1/60 秒的帧
lag_1_30：相隔 1/30 秒的帧

对 60 FPS 视频：

native       = t → t+1
lag_1_60     = t → t+1
lag_1_30     = t → t+2

但 Cadence motion gate 当前传入的是：

nr_raw_diff_1_60

并把它称作“较长时间尺度运动”。

对 60 FPS 视频，它根本不是较长尺度，而就是原生相邻帧。

正确设计应该是：

输入 FPS	Native cadence	Parent/common cadence
60 FPS	1/60 秒	1/30 秒
120 FPS	1/120 秒	1/60 秒

所以在 60 FPS 下，Cadence gate 应使用 1/30 的运动证据，而不是 1/60。

2. 低运动补偿误差被错误当成 Cadence 塌缩证据

当前 Cadence 风险把以下现象视为“没有新内容”：

MCT residual 很低
self composition error 很低
self cycle error 很低

并且两个信号一致就可以产生很高风险。

但对一段正常、连续、光流估计准确的战斗视频：

运动越平滑
光流越准确
MCT / composition / cycle residual 越低

这些本来也是高质量运动的表现。

只有在存在明显的奇偶帧信息不对称时，低 residual 才可能支持“复制帧或帧率塌缩”的结论。它们不能独立作为塌缩证据。

3. Cadence 在高风险采样窗口上聚合

NR 先选择 Uniform + Risk 窗口，然后只在这些窗口上计算 Cadence。Balanced 默认风险窗口数量比均匀窗口还多。

战斗特效、镜头震动、闪光、粒子和遮挡本来就容易被 Risk Selector 选中。

随后 Cadence 使用：

max(P80, median)

聚合窗口风险。

因此它实际回答的接近：

在一批故意挑出的高风险战斗片段中，Cadence 风险有多高？

而不是：

整条视频有多少区域真正发生了帧率塌缩？

这会系统性放大战斗视频的 Cadence Risk。

4. 战斗特效天然会产生奇偶相位能量

当前 Phase 使用绝对帧索引的奇偶性，将视频拆成：

偶数帧 phase A
奇数帧 phase B

再比较锐度、边缘和交替能量。

对 30→60 的插帧视频，这种相位拆分可能有意义；因为一组可能是原帧，另一组可能是生成帧。

但对真实 60 FPS 战斗视频，以下内容也可能产生周期性奇偶差：

-技能特效闪烁；
-粒子隔帧生成；
-屏幕震动；

* Bloom 或曝光变化；
    -游戏内部动画采样频率；
    -编码 GOP 和量化变化。

所以原始视频也得到了很低的 phase_consistency=42.7。

⸻

三、对当前结果的临时使用规则

在代码整改前，建议按以下方式阅读本次报告。

原始视频

可信：
Temporal stability 88.4
部分可信：
Technical quality 56.3
需要谨慎：
Motion smoothness 37.3
Phase consistency 42.7
UI/text stability 27.7
当前无效：
Cadence risk 0.93
Cadence integrity 12.8
Overall 8.8
Affected 52.3%

一个正常原始战斗视频出现：

22 Issues
52.3% Affected
严重问题

本身已经可以作为负对照证明：当前诊断阈值和持续时间估计明显过于激进。

插帧视频

相对于原始视频，最值得检查的是：

Phase consistency：42.7 → 24.5
Technical quality：56.3 → 49.0

建议打开相应 Issue 的 Compare Clip，逐帧查看：

-是否一帧清晰、一帧模糊；
-生成帧是否明显更软；
-细线、角色轮廓和技能特效是否隔帧变化；
-输出编码是否只对生成帧产生更严重的压缩；
-是否存在原帧和生成帧不同的锐化或降噪处理。

由于 Motion 和 Temporal 基本相同，目前没有证据证明插帧让运动连续性显著变差。

⸻

四、下一步评分逻辑的具体整改方案

P0：重写 60/120 FPS Cadence

这是最优先的修改。

1. 使用 FPS 自适应 Parent Lag

新增：

native_dt = median(diff(pts))
parent_dt = 2.0 * native_dt

对应：

60 FPS  → parent lag = 1/30
120 FPS → parent lag = 1/60

新增特征：

nr_raw_diff_native
nr_raw_diff_parent
nr_parent_motion_p90
nr_parent_moving_pixel_fraction

不要再固定使用 nr_raw_diff_1_60 作为所有 FPS 的 motion gate。

2. Cadence 必须以奇偶信息不对称为硬门槛

建议在移动区域计算相邻帧差：

d_t =
\operatorname{mean}_{x\in M_t}
|Y_{t+1}(x)-Y_t(x)|

父尺度运动：

p_t =
\operatorname{mean}_{x}
|Y_{t+2}(x)-Y_t(x)|

然后计算：

even median difference
odd median difference
phase asymmetry
phase coherence
moving duplicate fraction

例如：

A =
\frac{|\operatorname{median}(d_{\text{even}})
-\operatorname{median}(d_{\text{odd}})|}
{\operatorname{median}(d_{\text{even}})
+\operatorname{median}(d_{\text{odd}})+\epsilon}

Cadence Risk 只有在以下条件同时成立时才能大于零：

parent-scale motion 足够大
AND 奇偶相邻差明显不对称
AND 不对称方向在多个窗口内稳定
AND 某一相位缺少新的时序信息

低 MCT、低 composition、低 cycle 只能作为辅助证据，不能作为主要证据。

3. 增加 Phase Coherence

真实战斗特效可能在局部产生奇偶差，但这种差异通常：

-方向不稳定；
-只出现在少量区域；
-不同时间段相位会变化。

插帧输出的原帧/生成帧差异通常在整段视频中保持同一个相位。

应记录：

phase_gap_signed
phase_coherence
phase_coverage
dominant_phase

例如：

C =
\frac{
\left|\sum_i w_i\,g_i\right|
}{
\sum_i w_i|g_i|+\epsilon
}

其中 g_i 是带正负号的 phase gap。

没有高 coherence 时，不允许触发严重 Cadence 惩罚。

4. Cadence 应从全片低分辨率扫描估计

不要再从 Risk-selected windows 估计全片 Cadence。

建议在 Tier-1 低分辨率扫描中直接维护：

native adjacent differences
parent-lag differences
even/odd phase statistics
moving-pixel fractions
scene ID

风险窗口只用于：

-生成 Issue；
-输出 Heatmap；
-做精细诊断。

全局 Cadence 应从全片或每个场景的均匀序列统计。

⸻

五、暂时取消当前指数 Cadence 乘法

当前：

S = S_{\text{base}}\exp(-2.2R)

当 R=0.93 时，倍率只有约 0.13。

对于一个尚未完成真实视频标定的 NR Cadence 检测器，这个惩罚强度过于危险。

推荐的短期输出

在 Cadence v2 完成前：

artifact_quality = 69.1
cadence_integrity = diagnostic_only
overall_score = artifact_quality

报告显示两条独立轴：

字段	含义
Artifact Quality	画面、运动、技术和结构风险
Cadence Integrity	有效新增帧和奇偶相位风险
Final Quality	暂不提供或标记 uncalibrated

更保守的替代方案是最多只允许 Cadence 扣 15 分：

S_{\text{final}}
=
S_{\text{base}}
\left(0.85+0.15\frac{S_{\text{cadence}}}{100}\right)

但在完成真实视频标定前，分开报告比继续构造单一总分更诚实。

⸻

六、Phase Consistency 的具体整改

当前原始与插帧的主要差异完全由 Phase 驱动，因此必须提高它的解释能力。

增加 Two-phase Applicability

先判断视频是否真的具有稳定的双相位结构：

source_phase_likelihood
phase_coherence
phase_sharpness_direction
phase_edge_direction

只有当：

source_phase_likelihood >= 0.7
phase_coherence >= 0.6

时，Phase Consistency 才进入总分。

否则：

phase_consistency = N/A
phase_weight = 0

真实 60 FPS 原始视频通常不应被强制解释为“原帧相位 + 生成帧相位”。

提供可解释字段

报告中增加：

{
  "phase": {
    "applicable": true,
    "phase_a_sharpness": 312.4,
    "phase_b_sharpness": 205.7,
    "sharpness_ratio": 0.66,
    "phase_a_edge_density": 0.083,
    "phase_b_edge_density": 0.061,
    "coherence": 0.84,
    "dominant_bad_phase": "B"
  }
}

Issue Clip 应展示：

phase A frame | phase B frame | difference/edge map

而不是笼统地说“生成帧模糊”，因为 NR 模式实际上不知道哪一组一定是生成帧。

⸻

七、Motion Smoothness 对战斗特效的鲁棒性整改

原始视频只有 37.3，说明当前 Motion 分数也明显偏低。

战斗特效会导致：

-粒子出现和消失；
-透明叠加；
-大面积闪光；
-非刚体能量扩散；
-遮挡和显露；
-低纹理 Bloom；
-光流估计失效。

这些情况应先被判定为“光流证据不可靠”，而不是直接判定为运动不平滑。

增加 Flow Reliability Gate

每个窗口记录：

flow_valid_fraction
forward_backward_consistency
photometric_support_fraction
persistent_track_fraction
effect_transient_fraction

只有可靠区域进入：

fold
divergence
curl
acceleration
jerk
reversal

若有效覆盖不足，例如：

flow_valid_fraction < 0.25

则：

motion_smoothness = N/A 或低置信度

而不是给出 37 分。

区分运动错误和外观变化

新增两类：

trackable_motion_error
appearance_change_uncertainty

粒子、爆炸、闪光应主要增加 uncertainty，而不是直接增加 motion error。

验收标准应是：

真实 60 FPS 战斗视频：
motion_smoothness ≥70
或 flow coverage 不足时显示 N/A

不能继续稳定输出 30～40 分。

⸻

八、UI/Text 分支需要增加可靠性门控

两个视频都是约 27.8，说明 UI 分支没有提供比较价值。

当前战斗特效可能被 UI detector 误认为：

-屏幕固定边缘；
-文字笔画；
-持久组件；
-静态 HUD。

建议新增：

ui_detection_confidence
ui_persistence_seconds
ui_screen_motion
ui_component_area_ratio
ui_transient_rejection

只有满足：

长时间屏幕坐标固定
组件面积合理
跨帧持续存在
不属于全屏闪光或粒子

才进入 UI 分数。

动态但合法的内容，例如：

-技能冷却数字；
-血条变化；
-伤害数字；
-技能亮起；
-状态图标变化；

不能仅凭变化就判为不稳定，应区分：

内容更新
几何漂移
轮廓破损
双重曝光

⸻

九、Issues 和 Affected 比例需要重写

原始视频：

22 issues
52.3% affected

插帧视频：

19 issues
77.8% affected

这两个结果明显不可信。

当前同类 Issue 在间隔不超过 0.5 秒时会被聚合成一个长区间，然后 Affected 直接计算这些合并区间的并集。

这会把未实际采样和未确认的问题间隙也算成 affected。

正确设计

Issue Card 的合并区间和受影响时长必须分开。

每个 Issue 保存：

{
  "display_span": [10.0, 12.5],
  "support_spans": [
    [10.02, 10.09],
    [10.51, 10.58],
    [12.31, 12.38]
  ]
}

其中：

* display_span 用于将相近问题合并成一张卡；
* support_spans 才用于计算受影响时长；
    -未采样的 0.4 秒间隙不能自动计入。

更好的方案是使用 Tier-1 全片扫描估计 prevalence：

sampled_issue_fraction
estimated_affected_fraction
confirmed_affected_seconds

分开呈现。

⸻

十、Quality Level 需要拆分

当前页面显示“严重问题”，容易被理解为整条视频质量严重不合格。

应拆成：

Global Quality Level
Worst Local Issue Level

例如：

Global Quality：中等 / 未标定
Worst Local Issue：严重

一段 60 秒视频中存在一个严重局部 Issue，不等于整条视频是“严重问题”。

⸻

十一、针对这两条视频建立最小真实回归集

这两条视频非常适合作为下一阶段负对照和差异对照。

建议截取一段包含：

-大量战斗特效；
-角色运动；
-镜头运动；

* UI；
    -文字；
    -粒子和遮挡；

长度约 10～20 秒的片段。

构造以下版本：

版本	目标
True 60 FPS original	负对照
30 FPS 下采样后重复帧升到 60	明确 Cadence collapse
30→60 线性混合	模糊插帧
当前模型 30→60 或 60→60 输出	实际候选
原始 60 加压缩/模糊	技术质量对照

Cadence 验收

视频	目标
True 60 original	risk ≤0.10
Duplicate 30→60	risk ≥0.70
Linear blend 30→60	risk 0.20～0.60
好的插帧 60	risk ≤0.30

其他验收

原始视频 Affected ≤10%
原始视频不得出现全局 Severe
原始 Motion ≥70 或显示 N/A
原始 UI ≥70 或显示 N/A
插帧 Phase 应低于原始，但差异应能由 Phase A/B 证据解释

⸻

十二、当前最合适的评测方式

如果两条 60 FPS 视频是同一内容、逐帧时间对齐，并且原始视频是真实 60 FPS Ground Truth，不应只运行两个独立 NR。

应该运行：

rr-vfiqa inspect \
  --reference original_60.mp4 \
  --candidate interpolated_60.mp4 \
  --out runs/full_reference \
  --flow-backend farneback \
  --device cpu

自动路由应进入 full-reference，直接比较对应帧。Full Reference 比两个独立 NR 更适合判断插帧帧是否正确。

如果实际插帧输入是 30 FPS、输出是 60 FPS，则应该使用：

rr-vfiqa inspect \
  --reference source_30.mp4 \
  --candidate interpolated_60.mp4 \
  --out runs/endpoint

此时进入 Endpoint 模式。

NR 更适合：

只有一条视频时的风险筛查

不适合作为有 Ground Truth 或端点参考时的主要模型评价手段。项目文档也明确说明三种模式使用不同标尺，不能跨模式或脱离输入契约解释。

⸻

最终优先级

P0：立即修正

1. 60 FPS Cadence parent lag 改为 1/30。
2. 低 MCT/comp/cycle 不再独立构成 Cadence Risk。
3. Cadence 增加 phase asymmetry 和 phase coherence 硬门控。
4. Cadence 从全片均匀扫描统计，不使用风险窗口分布。
5. 暂停 Cadence 指数乘法，分别报告 Artifact Quality 和 Cadence。
6. Phase 增加双相位适用性判断。

P1：真实战斗视频鲁棒性

1. Motion 增加 Flow Reliability Gate。
2. 粒子、闪光和外观变化不直接作为运动错误。
3. UI 增加检测置信度和动态内容区分。
4. Affected 改为 support spans 或全片 prevalence。
5. Global Quality 与 Worst Issue 分离。

P2：效果验证

1. 将这条原始战斗视频作为负对照。
2. 建立 True 60、Duplicate 60、Blend 60、Model 60 梯度。
3. 使用 FR 或 Endpoint 结果标定 NR Phase/Cadence。
4. 达成原始视频 Cadence Risk ≤0.1 后，再恢复单一 Overall。

⸻

当前这两份报告最可靠的结论不是“8.8 对 8.0”，而是：

两条视频在普通时序和运动指标上几乎相同；插帧视频主要出现了更明显的奇偶相位质量差异和少量技术质量下降。当前 Cadence、UI、Affected 和全局严重等级存在明显误判，必须先整改后才能用于模型质量门禁。
