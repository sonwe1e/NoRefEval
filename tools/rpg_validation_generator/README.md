# rpg_validation_generator — 程序化 RPG 验证数据生成器

完全由 Python + OpenCV 代码绘制的 **RPG 场景验证数据生成器**，为 NoRefEval 的三种
评测模式（`no-reference` / `endpoint-2x` / `full-reference`）各生成 5 个带**精确已知缺陷**
的候选视频，并附带 source / reference / oracle、缺陷 mask、manifest 与 provenance。

完整规格见仓库根目录 [`PICPLAN.md`](../../PICPLAN.md)。

> ⚠️ **数据性质声明**：目录虽保留 `nr_real` / `endpoint_real` / `fr_real` 之名，但每个
> manifest 都明确声明 `"data_origin": "procedural_synthetic_rpg"`、`"real_capture": false`、
> `"production_gate": false`。这是**程序化合成数据**，不是真实录制，也不是 AIGC 产物，
> 不能用于生产上线门禁。本工具不运行 NoRefEval、不调分、不训练、不联网、不下载素材。

---

## 依赖

仅需 `numpy`、`opencv-python(-headless)`、`av`（PyAV）与 `pytest`（测试），均在仓库
`pyproject.toml` 的基础依赖中。视频编码用 PyAV（H.264 / yuv420p / 无音频）。FFmpeg 仅用于
provenance 记录版本，非必需。所有纹理 / 角色 / 武器 / UI / 文字均由代码绘制（OpenCV Hershey
字体，无外部字体文件）。

---

## 生成

正式生成（1280×720 / 4 s / 120 FPS，PICPLAN §3）：

```bash
python -m tools.rpg_validation_generator generate \
  --output validation --resolution 1280x720 --duration 4 \
  --master-fps 120 --seed 20260729 --mode all
```

调试 / 快速生成（640×360 / 2 s）：

```bash
python -m tools.rpg_validation_generator generate --output validation_draft --draft --mode all
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--mode {all,nr,endpoint,fr}` | 选择模式 |
| `--draft` | 640×360 / 2 s 开发预设 |
| `--resolution WxH` | 覆盖分辨率（同时按比例调世界画布） |
| `--duration` / `--master-fps` / `--seed` | 覆盖时长 / 主帧率 / 随机种子 |
| `--overwrite` | 删除已存在的目标模式目录后再生成 |
| `--workdir` | 指定 master memmap 暂存目录（默认临时目录，结束自动清理） |

---

## 校验

```bash
python -m tools.rpg_validation_generator validate --root validation
```

返回 0 表示通过。校验内容（PICPLAN §19）：每个视频可被 PyAV 打开、FPS / 帧数 / 分辨率与
manifest 一致、无音频流、缺陷区间在时长内、文件 SHA256 匹配；endpoint 偶数帧契约
（`candidate[0::2] == source`，编码前精确成立，编码后用容差复核并以 `raw_rgb_sha256` 精确佐证）；
FR 的 reference/candidate 同帧同分辨率且缺陷 mask 外一致；NR 的 candidate 帧率分布为
2×60 + 3×120。

---

## 输出结构

```text
validation/
├── nr_real/
│   ├── case_01/ … case_05/
│   │   ├── candidate_{60|120}.mp4
│   │   ├── oracle_clean_{60|120}.mp4
│   │   ├── defect_masks/        # 320×180 缺陷 footprint（PICPLAN §13）
│   │   ├── _contract_masks/     # 全分辨率契约 mask（供精确校验）
│   │   ├── manifest.json
│   │   └── provenance.json
│   └── manifest.jsonl
├── endpoint_real/   # source_60 / candidate_120 / oracle_gt_120
└── fr_real/         # source_30 / reference_60 / candidate_60
```

每个 case 含 5 个复用场景之一（草原追击 / 遗迹遮挡 / Boss 挥剑 / 城镇商店 / 魔法战斗），
缺陷在 master 生成后由独立 `DefectOperator` 注入（绝不藏进场景渲染代码）。

---

## 代码结构

```text
tools/rpg_validation_generator/
├── config.py / schema.py / motion.py / case_specs.py   # 参数 / 契约 / 运动 / 15 case 表
├── scenes.py                                            # 5 个 RPG 场景
├── renderer/   primitives, camera, tilemap, character, effects, ui, renderer
├── timeline.py / encode.py / hashing.py                 # master memmap / 编码 / 哈希
├── defects/    base + blur/freeze/cadence/ghost/shift/delete/ui/flicker
├── builder.py / orchestration.py                        # 单 case / 全流程编排
├── manifests.py / provenance.py / validate.py           # 清单 / 溯源 / 校验
└── __main__.py                                          # CLI
tests/test_rpg_generator_*.py                            # 6 个测试文件
```

`case_specs.py` 是 15 个 case 的唯一真值表，builder 与测试共用，保证契约同步。

---

## 测试

```bash
pytest tests/test_rpg_generator_determinism.py \
       tests/test_rpg_generator_sampling.py \
       tests/test_rpg_generator_defects.py \
       tests/test_rpg_generator_endpoint.py \
       tests/test_rpg_generator_fr.py \
       tests/test_rpg_generator_outputs.py
```

覆盖：相同 seed 的 raw RGB 完全一致；按帧索引采样（非 ffmpeg 滤镜）；缺陷局部化、窗口外不变、
endpoint 仅奇数帧、UI 漂移不改世界；endpoint 偶数帧编码前精确契约；FR 几何/帧率同构与 mask 外一致；
全树 `validate` 通过 + CLI 往返 + origin 标志 + NR 帧率分布 + 全无音频。测试用 320×180 / 2 s 小配置。

---

## 确定性

所有运动是 `(seed, frame, t, 固定参数)` 的解析函数；随机量在场景 setup 阶段一次性从 seeded RNG
抽取（粒子池 / 道具位置），帧循环内不再消费 RNG。同配置重复生成的 `raw_rgb_sha256` 完全一致。

---

## 已知限制

- 角色 / 道具以几何图元组合绘制，刻意保持简单（非 sprite），用于覆盖运动 / 遮挡 / 细物体 /
  文字 / UI 等评测维度，而非追求美术质量。
- H.264 + yuv420p 编码会引入约 2–3 的平均像素误差（细边缘 / 文字色度抽样处局部更高）；因此
  **精确契约以编码前的 raw 帧与 `raw_rgb_sha256` 为准**，编码后校验使用容差。
- `_contract_masks/` 为全分辨率 PNG，体积较大，仅供校验；320×180 的 `defect_masks/` 是清单引用
  的展示 mask。两者都不应被当作“真实标注”。
- 仓库根 `.gitignore` 忽略 `*.mp4`，故生成的视频默认不入库；本工具负责**生成**，数据按需在本地
  产出。
