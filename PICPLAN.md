# RPG 程序化视频验证数据生成计划

## 1. 项目目标

实现一个**完全由 Python 代码程序化生成**的 RPG 游戏视频数据生成器，为 NoRefEval 的三种评测模式生成小规模验证数据：

```text
validation/nr_real/
validation/endpoint_real/
validation/fr_real/
```

本项目只负责：

1. 生成程序化 RPG 场景。
2. 生成无缺陷的标准时间线。
3. 生成带有精确已知缺陷的候选视频。
4. 生成每种模式需要的 source、reference、candidate 和 oracle 视频。
5. 输出缺陷类型、时间段、ROI、随机种子和文件哈希。
6. 提供数据生成与数据契约测试。

本项目**不负责**：

* 运行 NoRefEval。
* 计算评测分数。
* 训练或标定评测器。
* 接入真实插帧模型。
* 实现人工 A/B 标注系统。
* 使用任何 AIGC 图像或视频模型。
* 下载网络素材。
* 调用 Qwen、Wan、Stable Diffusion 或其他生成式模型 API。
* 将程序化数据描述为真实录制数据。

虽然目录保留 `nr_real`、`endpoint_real` 和 `fr_real`，所有 manifest 必须明确声明：

```json
{
  "data_origin": "procedural_synthetic_rpg",
  "real_capture": false,
  "production_gate": false
}
```

---

## 2. 数据规模

“每种模式 5 个视频”定义为：

> 每种模式生成 5 个正式 candidate 视频。

三种模式总计生成：

```text
No-Reference candidate：5 个
Endpoint-2x candidate：5 个
Full-Reference candidate：5 个
```

source、reference、oracle 和 master 视频属于辅助文件，不计入这 15 个 candidate 视频。

五个基础 RPG 场景在三种模式中复用，避免重复设计，同时保证三种模式覆盖相同的内容和运动条件。

---

## 3. 固定生成参数

默认生成参数：

```text
分辨率：1280 × 720
场景时长：4 秒
Master FPS：120
Master 帧数：480
颜色格式：RGB uint8
随机种子基值：20260729
```

开发调试时允许：

```text
分辨率：640 × 360
场景时长：2 秒
```

正式生成必须使用 1280×720、4 秒和 120 FPS。

所有场景必须通过解析公式或确定性关键帧计算运动，不允许依赖实时物理模拟产生不可复现结果。

---

## 4. 技术栈

只允许使用：

```text
Python 3.10+
NumPy
OpenCV
Pillow（可选）
PyAV 或 FFmpeg 命令行
pytest
标准库
```

优先使用 OpenCV 绘制：

* 多边形。
* 圆形。
* 椭圆。
* 线段。
* 矩形。
* 粒子。
* 文字。

文字使用 OpenCV Hershey 字体，不依赖外部字体文件。

所有纹理、人物、武器、建筑、UI 和背景必须由代码生成，不允许读取下载的外部美术素材。

---

## 5. 总体数据流程

```text
SceneSpec
    ↓
程序化 RPG Renderer
    ↓
clean_master_120
    ↓
时间采样与缺陷注入
    ↓
NR candidate
Endpoint source/candidate/oracle
FR source/reference/candidate
    ↓
视频编码
    ↓
manifest + defects + provenance + hashes
```

Master 视频是所有模式的唯一干净时间真值。

任何缺陷都必须在 Master 视频生成完成后，通过独立 defect operator 注入，不允许把缺陷隐藏在场景渲染代码中。

---

## 6. 程序化 RPG 渲染器

### 6.1 场景空间

使用 2D/2.5D RPG 场景。

场景包含两个坐标空间：

```text
world space：地图、角色、敌人、武器、建筑、粒子
screen space：血条、技能按钮、摇杆、文字、商店、对话框
```

Camera 只作用于 world-space 内容，不作用于 screen-space UI。

建议先在大于画面的 world canvas 上绘制，例如：

```text
world canvas：1600 × 1000
viewport：1280 × 720
```

再通过 affine transform 生成最终视口。

### 6.2 图层顺序

建议图层顺序：

```text
background
ground decoration
rear building
character/enemy
weapon/projectile
foreground occluder
world text
screen UI
post effects
```

所有图层必须使用明确的 z-order。

### 6.3 公共绘制对象

至少实现：

```text
TileMap
Character
Enemy
NPC
Sword
Projectile
ParticleEmitter
Pillar
Tree
Rock
HealthBar
SkillButton
Joystick
CooldownNumber
FloatingName
DialogPanel
ShopPanel
```

角色可以使用圆、椭圆、多边形和线段组合，不需要复杂 sprite。

### 6.4 公共运动函数

至少实现以下确定性运动函数：

```python
linear(t)
smoothstep(t)
ease_in_out_cubic(t)
bezier_2d(t, p0, p1, p2, p3)
sinusoidal(t, frequency, amplitude)
```

所有物体位置、旋转、缩放和透明度必须由：

```text
scene seed
frame index
时间 t
固定参数
```

唯一决定。

---

## 7. 五个基础 RPG 场景

所有模式复用以下五个场景。

### Scene 01：草原追击

内容：

* 玩家从左向右奔跑。
* 一个敌人在后方追击。
* Camera 水平跟随。
* 地面包含草地、石块和道路。
* 左下角显示移动摇杆。
* 右下角显示三个技能按钮。
* 顶部显示玩家血条。

主要覆盖：

```text
匀速运动
轻微加速
Camera 平移
人物轮廓
固定 HUD
```

### Scene 02：遗迹遮挡

内容：

* 玩家从遗迹走廊中穿过。
* 前景有两根柱子遮挡角色。
* Camera 同时发生小幅旋转和缩放。
* 角色从柱子后方重新显露。
* 地面有不同深度的装饰层。

主要覆盖：

```text
Camera 旋转
Camera 缩放
遮挡
显露背景
多层运动
```

### Scene 03：Boss 挥剑

内容：

* 玩家接近 Boss。
* 玩家执行快速剑击。
* 剑是长度较长、宽度较小的细物体。
* Boss 执行后退和转向。
* 命中时出现短暂闪光。
* 剑的角度和位置必须连续变化。

主要覆盖：

```text
细物体
快速旋转
方向变化
人物局部运动
短时高频动作
```

### Scene 04：城镇 NPC 与商店

内容：

* 玩家缓慢走向 NPC。
* NPC 上方显示浮动名字。
* 屏幕中下方出现对话框。
* 右侧弹出商店面板。
* 商店中有多个程序化图标和价格文字。
* 技能冷却数字发生单调变化。

主要覆盖：

```text
文字
浮动名称
静态 UI
动态 UI
弹窗
组件数量变化
```

### Scene 05：魔法战斗

内容：

* 玩家释放一个魔法弹。
* 魔法弹沿曲线路径飞行。
* 命中时生成粒子和冲击圆环。
* Camera 发生短时屏幕震动。
* 敌人血条下降。
* 技能按钮进入冷却。

主要覆盖：

```text
快速粒子
曲线运动
局部闪光
Camera 震动
血条变化
技能冷却
```

---

## 8. Master 视频和时间采样

每个场景首先生成：

```text
master_clean_120.mp4
```

Master 时间点为：

```text
t_i = i / 120
i = 0 ... 479
```

从 Master 派生：

```text
clean_120：全部帧
clean_60：Master[0, 2, 4, ...]
clean_30：Master[0, 4, 8, ...]
```

禁止通过 FFmpeg 帧率转换滤镜生成 60 FPS 或 30 FPS。

必须在 Python 中按帧索引采样，确保时间关系完全明确。

---

## 9. No-Reference 数据

目录：

```text
validation/nr_real/
├── case_01/
├── case_02/
├── case_03/
├── case_04/
├── case_05/
└── manifest.jsonl
```

每个 case 正式发布一个 candidate 视频，并保存一个隐藏 oracle clean 视频供后续验证使用。

### NR Case 01：60 FPS Freeze

来源场景：

```text
Scene 01
```

输出：

```text
candidate_60.mp4
oracle_clean_60.mp4
```

缺陷：

```text
类型：freeze
时间段：1.50 ～ 2.20 秒
操作：该时间段内连续复制缺陷开始前一帧
```

### NR Case 02：120 FPS 奇数帧模糊

来源场景：

```text
Scene 02
```

输出：

```text
candidate_120.mp4
oracle_clean_120.mp4
```

缺陷：

```text
类型：odd_frame_blur
时间段：1.20 ～ 2.80 秒
操作：仅对该时间段中的奇数帧执行 Gaussian Blur
kernel：15×15
sigma：3.0
```

### NR Case 03：120 FPS Cadence Collapse

来源场景：

```text
Scene 03
```

输出：

```text
candidate_120.mp4
oracle_clean_120.mp4
```

缺陷：

```text
类型：cadence_collapse
时间段：1.40 ～ 2.70 秒
操作：Y[2i+1] = Y[2i]
```

该视频在容器中仍为 120 FPS，但缺陷时间段中的有效内容采样率下降为约 60 FPS。

### NR Case 04：60 FPS UI Drift

来源场景：

```text
Scene 04
```

输出：

```text
candidate_60.mp4
oracle_clean_60.mp4
```

缺陷：

```text
类型：ui_drift
时间段：1.00 ～ 3.10 秒
操作：
技能区每帧在 x/y 方向进行 ±2～4 像素周期性漂移
浮动名字发生 1 像素奇偶摆动
世界画面保持不变
```

### NR Case 05：120 FPS Ghost/Flicker

来源场景：

```text
Scene 05
```

输出：

```text
candidate_120.mp4
oracle_clean_120.mp4
```

缺陷：

```text
类型：projectile_ghost_flicker
时间段：1.30 ～ 2.60 秒
操作：
奇数帧将魔法弹当前帧与前一帧以 0.5/0.5 混合
每四帧降低一次粒子 alpha
仅作用于 projectile/particle ROI
```

---

## 10. Endpoint-2x 数据

目录：

```text
validation/endpoint_real/
├── case_01/
├── case_02/
├── case_03/
├── case_04/
├── case_05/
└── manifest.jsonl
```

每个 case 包含：

```text
source_60.mp4
candidate_120.mp4
oracle_gt_120.mp4
```

派生关系：

```text
source_60[i] = master_120[2i]
oracle_gt_120 = master_120
```

Endpoint candidate 必须严格满足：

```text
candidate_120[2i] = source_60[i]
```

所有缺陷只能修改奇数帧。

### Endpoint Case 01：运动模糊

来源：

```text
Scene 01
```

缺陷：

```text
类型：generated_motion_blur
时间段：1.20 ～ 2.50 秒
操作：只模糊奇数帧中的玩家和敌人 ROI
```

### Endpoint Case 02：显露区域 Ghost

来源：

```text
Scene 02
```

缺陷：

```text
类型：disocclusion_ghost
时间段：1.30 ～ 2.70 秒
操作：
只修改奇数帧
在柱子边缘和角色显露区域混入前一端点内容
```

### Endpoint Case 03：剑位置错误

来源：

```text
Scene 03
```

缺陷：

```text
类型：thin_weapon_wrong_motion
时间段：1.10 ～ 2.40 秒
操作：
只修改奇数帧
剑的角度增加周期性 ±8 度误差
剑尖位置增加 3～6 像素偏移
```

### Endpoint Case 04：UI/Text 错位

来源：

```text
Scene 04
```

缺陷：

```text
类型：ui_text_drift
时间段：1.00 ～ 3.00 秒
操作：
只修改奇数帧
商店面板水平偏移 3 像素
价格文字发生 1 像素奇偶抖动
```

### Endpoint Case 05：Freeze-Copy 与粒子重影

来源：

```text
Scene 05
```

缺陷：

```text
类型：generated_freeze_copy
时间段：1.40 ～ 2.50 秒
操作：
部分奇数帧复制前一个偶数端点
其余奇数帧的粒子与前一端点进行混合
```

### Endpoint 强制契约

生成后必须在原始 RGB 帧上验证：

```python
np.array_equal(candidate_frames[0::2], source_frames)
```

编码后允许轻微压缩误差，但编码前必须完全一致。

---

## 11. Full-Reference 数据

目录：

```text
validation/fr_real/
├── case_01/
├── case_02/
├── case_03/
├── case_04/
├── case_05/
└── manifest.jsonl
```

每个 case 包含：

```text
source_30.mp4
reference_60.mp4
candidate_60.mp4
```

派生关系：

```text
reference_60[i] = master_120[2i]
source_30[i] = master_120[4i]
```

Candidate 与 Reference 必须具有：

```text
相同 FPS
相同帧数
相同分辨率
相同 PTS 语义
```

### FR Case 01：全局模糊

来源：

```text
Scene 01
```

缺陷：

```text
类型：global_blur
时间段：1.20 ～ 2.60 秒
kernel：21×21
sigma：5.0
```

### FR Case 02：局部空间偏移

来源：

```text
Scene 02
```

缺陷：

```text
类型：local_spatial_shift
时间段：1.30 ～ 2.70 秒
ROI：角色和柱子交界区域
操作：ROI 水平移动 4 像素
```

### FR Case 03：细物体删除

来源：

```text
Scene 03
```

缺陷：

```text
类型：thin_object_delete
时间段：1.10 ～ 2.40 秒
操作：删除部分剑身和剑尖
背景使用邻域颜色填补
```

### FR Case 04：UI/Text 错误

来源：

```text
Scene 04
```

缺陷：

```text
类型：ui_text_corruption
时间段：1.00 ～ 3.00 秒
操作：
商店面板偏移 3 像素
价格文字笔画变粗
技能按钮增加轻微颜色偏移
```

### FR Case 05：时序冻结与亮度闪烁

来源：

```text
Scene 05
```

缺陷：

```text
类型：temporal_freeze_flicker
时间段：1.40 ～ 2.60 秒
操作：
连续复制 3～4 帧
随后对局部战斗区域添加 ±12 亮度交替
```

每个 FR case 只设置一个主要缺陷主题，避免无法判断哪个指标应该响应。

---

## 12. 缺陷注入架构

所有缺陷继承统一接口：

```python
class DefectOperator:
    defect_type: str

    def apply(
        self,
        frames: np.ndarray,
        fps: int,
        start_time: float,
        end_time: float,
        context: DefectContext,
    ) -> DefectResult:
        ...
```

`DefectResult` 至少包含：

```python
@dataclass
class DefectResult:
    frames: np.ndarray
    affected_indices: list[int]
    affected_interval_seconds: tuple[float, float]
    roi_boxes: list[list[int]]
    mask_paths: list[str]
    parameters: dict
```

推荐实现文件：

```text
defects/
├── blur.py
├── freeze.py
├── cadence.py
├── ghost.py
├── spatial_shift.py
├── local_delete.py
├── ui_drift.py
├── text_corruption.py
└── flicker.py
```

---

## 13. ROI 和缺陷 Mask

渲染器应为每帧维护可选语义 mask：

```text
character
enemy
weapon
projectile
particle
ui
text
foreground_occluder
```

不需要把全部语义 mask 作为最终数据全部保存。

最终必须保存：

1. 缺陷的 bounding box。
2. 缺陷开始和结束帧。
3. 缺陷时间段。
4. 缺陷作用的语义类别。
5. 缺陷帧对应的低分辨率 mask。

缺陷 mask 保存为：

```text
320 × 180 PNG
```

只保存缺陷时间段中的 mask，避免数据量过大。

目录：

```text
case_xx/
└── defect_masks/
    ├── 000180.png
    ├── 000181.png
    └── ...
```

---

## 14. 视频编码

帧生成完成后，通过 FFmpeg 或 PyAV 编码。

正式 candidate 文件使用：

```text
容器：MP4
编码：H.264
pixel format：yuv420p
固定帧率
关闭音频
```

建议参数：

```bash
ffmpeg \
  -framerate 120 \
  -i frames/%06d.png \
  -c:v libx264 \
  -preset medium \
  -crf 10 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  output.mp4
```

Ground Truth 除 MP4 外，建议额外保存一个无损版本：

```text
FFV1 MKV
```

或者保存原始帧 SHA256。

视频编码可能因 FFmpeg 版本产生不同文件 hash，因此必须额外保存：

```text
raw_rgb_sha256
```

计算方法是将所有 RGB 帧按顺序拼接后更新 SHA256，不需要将原始数组全部写入单个文件。

---

## 15. Manifest 格式

每个 case 保存独立 `manifest.json`。

示例：

```json
{
  "schema_version": "rpg-procedural-validation-v1",
  "case_id": "endpoint_case_03",
  "mode": "endpoint-2x",
  "scene_id": "scene_03_boss_sword",
  "seed": 20260732,
  "data_origin": "procedural_synthetic_rpg",
  "real_capture": false,
  "production_gate": false,
  "resolution": [1280, 720],
  "duration_seconds": 4.0,
  "master_fps": 120,
  "files": {
    "source": "source_60.mp4",
    "candidate": "candidate_120.mp4",
    "oracle": "oracle_gt_120.mp4"
  },
  "defects": [
    {
      "type": "thin_weapon_wrong_motion",
      "start_time": 1.1,
      "end_time": 2.4,
      "start_frame": 132,
      "end_frame": 288,
      "semantic_targets": ["weapon"],
      "roi_boxes": [],
      "mask_directory": "defect_masks",
      "parameters": {
        "angle_error_degrees": 8,
        "tip_offset_pixels": 6
      }
    }
  ],
  "generator": {
    "name": "rpg_validation_generator",
    "version": "0.1.0",
    "commit": "unknown"
  },
  "hashes": {
    "candidate_sha256": "",
    "raw_rgb_sha256": "",
    "scene_spec_sha256": ""
  }
}
```

每个模式目录下再生成一个汇总：

```text
manifest.jsonl
```

每行对应一个 case。

---

## 16. Provenance

每个 case 保存：

```text
provenance.json
```

至少记录：

```json
{
  "python_version": "",
  "numpy_version": "",
  "opencv_version": "",
  "pyav_version": "",
  "ffmpeg_version": "",
  "generator_commit": "",
  "working_tree_dirty": null,
  "scene_seed": 0,
  "scene_spec_sha256": "",
  "raw_rgb_sha256": "",
  "video_sha256": {},
  "generation_command": ""
}
```

如果当前目录不是 Git 仓库：

```text
generator_commit = "unknown"
working_tree_dirty = null
```

不得伪造 commit。

---

## 17. 推荐代码结构

```text
tools/rpg_validation_generator/
├── __init__.py
├── __main__.py
├── config.py
├── schema.py
├── scenes.py
├── timeline.py
├── renderer/
│   ├── __init__.py
│   ├── primitives.py
│   ├── tilemap.py
│   ├── character.py
│   ├── effects.py
│   ├── camera.py
│   ├── ui.py
│   └── renderer.py
├── defects/
│   ├── __init__.py
│   ├── base.py
│   ├── blur.py
│   ├── freeze.py
│   ├── cadence.py
│   ├── ghost.py
│   ├── shift.py
│   ├── delete.py
│   ├── ui.py
│   └── flicker.py
├── builders/
│   ├── nr.py
│   ├── endpoint.py
│   └── full_reference.py
├── encode.py
├── manifests.py
├── provenance.py
├── validate.py
└── hashing.py
```

测试：

```text
tests/
├── test_rpg_generator_determinism.py
├── test_rpg_generator_sampling.py
├── test_rpg_generator_endpoint.py
├── test_rpg_generator_fr.py
├── test_rpg_generator_defects.py
└── test_rpg_generator_outputs.py
```

---

## 18. CLI

必须提供统一命令：

```bash
python -m tools.rpg_validation_generator generate \
  --output validation \
  --resolution 1280x720 \
  --duration 4 \
  --master-fps 120 \
  --seed 20260729 \
  --mode all
```

支持：

```text
--mode all
--mode nr
--mode endpoint
--mode fr
--draft
--overwrite
```

Draft 模式：

```bash
python -m tools.rpg_validation_generator generate \
  --output validation_draft \
  --draft \
  --mode all
```

校验命令：

```bash
python -m tools.rpg_validation_generator validate \
  --root validation
```

---

## 19. 自动校验

### 19.1 通用校验

每个视频必须满足：

* 可以被 PyAV 打开。
* FPS 与 manifest 一致。
* 帧数与 manifest 一致。
* 分辨率与 manifest 一致。
* PTS 单调。
* 无音频流。
* 缺陷时间段在视频范围内。
* 所有文件 SHA256 已写入 manifest。
* 相同 seed 和配置生成相同 `raw_rgb_sha256`。

### 19.2 NR 校验

* 共 5 个 candidate。
* 包含 2 个 60 FPS candidate。
* 包含 3 个 120 FPS candidate。
* oracle 与 candidate 在缺陷区间外一致。
* Cadence Collapse 只修改指定时间段中的奇数帧。
* UI Drift 不能修改 world-space RGB 区域。

### 19.3 Endpoint 校验

每个 case 必须满足：

```text
source FPS = 60
candidate FPS = 120
oracle FPS = 120
candidate frame count = 2 × source frame count
```

编码前严格验证：

```python
candidate_raw[0::2] == source_raw
```

缺陷只能影响奇数帧。

### 19.4 FR 校验

每个 case 必须满足：

```text
reference FPS = 60
candidate FPS = 60
source FPS = 30
reference frame count = candidate frame count
reference geometry = candidate geometry
```

Candidate 与 Reference 在缺陷时间段外必须完全一致，或者只允许由统一编码造成的微小误差。

---

## 20. 验收标准

最终交付必须满足：

1. 三种模式分别存在 5 个 candidate 视频。
2. 五个 RPG 场景均成功生成。
3. 所有视频都由代码绘制，不包含外部图片或 AIGC 内容。
4. 所有运行不需要网络。
5. 同一配置重复生成的原始 RGB hash 一致。
6. Endpoint 偶数帧契约在编码前完全成立。
7. FR reference/candidate 严格同帧对应。
8. 每个缺陷都有精确时间段和参数。
9. 每个 case 都有 manifest、provenance 和 hash。
10. `pytest` 全部通过。
11. `validate` 命令返回 0。
12. README 中给出生成、校验和目录说明。

---

## 21. 实现顺序

严格按照以下顺序实现。

### Phase 1：公共渲染器

实现：

* 配置和 SceneSpec。
* 基础图元。
* Camera。
* world-space 与 screen-space。
* 五个 RPG 场景。
* Master 120 FPS 生成。

验收：

```text
五个 clean master 可以稳定生成
相同 seed 的 raw RGB hash 一致
```

### Phase 2：采样和编码

实现：

* 120→60。
* 120→30。
* PNG/FFV1/MP4 输出。
* 视频 metadata 检查。

### Phase 3：缺陷注入器

依次实现：

```text
blur
freeze
cadence collapse
ghost
local shift
local delete
UI drift
text corruption
flicker
```

### Phase 4：三模式 Builder

实现：

```text
build_nr_cases()
build_endpoint_cases()
build_fr_cases()
```

### Phase 5：Manifest 和 Provenance

实现：

* 文件 hash。
* raw RGB hash。
* scene spec hash。
* Git 信息。
* FFmpeg 信息。
  -汇总 manifest.jsonl。

### Phase 6：测试与最终生成

完成 pytest 后，生成正式 1280×720 数据。

---

## 22. 明确禁止事项

实现过程中不得：

* 使用任何在线服务。
* 调用任何 AIGC 模型。
* 下载图片、视频或字体。
* 使用真实游戏名称、角色、Logo 或素材。
* 将合成数据写成 real capture。
* 接入 NoRefEval 并自动调分。
* 修改 NoRefEval 的指标或阈值。
* 实现人工 A/B。
* 实现模型训练或校准。
* 为增加复杂度而引入 Unity、Unreal、Blender 或浏览器渲染。
* 擅自将数据量从每种模式 5 个 candidate 扩大。
* 在没有测试的情况下修改 Endpoint 帧索引契约。

---

## 23. 最终交付内容

提交内容应包括：

```text
tools/rpg_validation_generator/    完整生成器
tests/                             完整测试
validation/nr_real/                5 个 NR candidate 及辅助文件
validation/endpoint_real/          5 个 Endpoint candidate 及辅助文件
validation/fr_real/                5 个 FR candidate 及辅助文件
PLAN.md                            本计划
README.md                          使用说明
```

最终总结中只需要报告：

```text
生成了哪些文件
各模式的视频数量
pytest 是否通过
validate 是否通过
正式生成命令
已知限制
```

不要在本任务中运行 NoRefEval，也不要根据生成的视频调整评测器。
