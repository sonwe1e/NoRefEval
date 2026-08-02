# 项目结构与环境

> 本仓库的代码结构、运行环境与开发约定总览。作为 README 的补充，README 面向「怎么用」，本文档面向「代码长什么样、跑在什么环境里、怎么改」。
>
> **规范：Python 统一使用 `G:\ds-torch\Scripts\python.exe`（见 [§2 环境](#2-环境python--ds-torch)）。**

- 适用版本：源码 `rr_vfiqa/_version.py` = **0.4.0**（以此为准；磁盘上另有旧元数据 0.1.0 / 0.3.0，见 §10）
- 上次整理：2026-08-02

---

## 1. 项目概览

`rr_vfiqa` 是一个 **模式感知的视频插帧质量评估** 工具：接收候选视频（以及可选的参考视频），自动判断能安全使用的评测条件，输出质量分数、置信度、问题时间段、热力图、关键帧和坏例视频。

三种评测模式使用**数学上互不相同的分数契约**，`overall_score` 禁止跨模式直接比较：

| 模式 | 输入 | 输出含义 | Schema |
|---|---|---|---|
| `no-reference` | 单条 60/120 FPS 视频 | 时序稳定性与伪影风险 | `nr-stability-risk-v4` |
| `endpoint-2x` | 60 FPS 端点参考 + 120 FPS 候选 | 端点约束下的插帧质量 | `endpoint-reduced-reference-v3` |
| `full-reference` | 逐帧对应的 60 FPS GT + 60 FPS 候选 | 同帧空间与时序保真度 | `fr-same-rate-fidelity-v3` |

状态：research prototype / metric development framework。公式融合未完成真实数据标定，`production_gate=false`。

入口命令（4 个子命令）：

```bash
python -m rr_vfiqa.cli evaluate       # 显式模式评分
python -m rr_vfiqa.cli compare        # 同源候选排序
python -m rr_vfiqa.cli inspect        # 单命令自动安全评测（自动路由模式）
python -m rr_vfiqa.cli inspect-batch  # 按 manifest 批量
```

---

## 2. 环境：Python = ds-torch

### 2.1 规范解释器

| 命令 | 实际路径 | 版本 | 是否可用 |
|---|---|---|---|
| `python` | `G:\ds-torch\Scripts\python.exe` | Python 3.12.13（uv 管理） | ✅ **唯一使用它** |
| `python3` | `C:\msys64\ucrt64\bin\python3.exe` | Python 3.14.5 | ❌ 切勿使用 |
| `py` | 未安装 | — | ❌ |
| `.venv\Scripts\python.exe` | — | （坏） | ❌ |

**规则：一切命令用 `python`（即 ds-torch），绝不写 `python3`。** 仓库源码树里混有 `cpython-314.pyc`，说明历史上有人在 3.14 下跑过，3.14 不是受支持的解释器。

### 2.2 `.venv` 已损坏，不要用

项目根 `.venv` 创建时基于 `C:\Users\Sonwe\miniforge3\python.exe`（Python 3.12.8），该基础解释器已被移除，所以 `.venv` 下所有可执行文件（`python.exe` / `pip` / `pytest` / `rr-vfiqa` / `pyav` …）都会直接报错：

```
No Python at '"C:\Users\Sonwe\miniforge3\python.exe'
```

`.venv` 目录本身被 gitignore，可以整体删除或重建（见 §2.5）。

### 2.3 导入模型：未 pip 安装，靠 cwd / PYTHONPATH

`rr_vfiqa` **没有**安装在 ds-torch 环境里（`G:\ds-torch` 里没有 editable install）。控制台脚本 `rr-vfiqa` 只存在于损坏的 `.venv` 里，**不能**直接敲 `rr-vfiqa`。

两种可用方式：

```bash
# 方式 A：在仓库根目录运行（推荐，cwd 可导入）
cd I:\WorkStations\NoRefEval
python -m rr_vfiqa.cli inspect --candidate out.mp4 --out runs/nr

# 方式 B：设置 PYTHONPATH
PYTHONPATH=I:\WorkStations\NoRefEval python -m rr_vfiqa.cli --help
```

测试同理，在仓库根目录：

```bash
python -m pytest                       # CPU Farneback 路径（默认）
RR_VFIQA_TEST_FLOW=raft python -m pytest   # GPU RAFT 路径
```

### 2.4 ds-torch 环境里已装 / 缺失的依赖

已装（核心均可用）：`numpy 2.4.4`、`opencv 4.13.0`、`av 18.0.0`、`scipy 1.17.1`、`torch 2.13.0.dev20260504+cu132`（CUDA 可用）、`lightgbm`、`matplotlib 3.10.9`、`pytest 9.0.3`。

**缺失的 extras**（用了会失败）：

| extra | 依赖 | 状态 |
|---|---|---|
| `vqa` | `pyiqa>=0.1.13` | ❌ 未安装 |
| `audit` | `cotracker>=0.3` | ❌ 未安装 |

即 Tier-3 audit 追踪器（`models/tracker_backend.py` 的 CoTracker）和 NR 的 pyIQA/NIQE 弱先验在当前环境不可运行；运行时若有 audit 预设会回退到 KLT。

### 2.5 从零重建环境（如需）

```bash
# 用 uv（与 ds-torch 一致的管理器）或 python -m venv 均可
uv venv .venv
uv pip install -e ".[dev,fusion,viz]"
# 需要 GPU 光流再加 torch；需要 vqa/audit 再加对应 extra
```

依赖策略：**只有 `pyproject.toml` 的 extras，无 requirements.txt / 无 lockfile**。声明的最小依赖：`numpy>=1.24, opencv-python-headless>=4.8, scipy>=1.10, av>=11.0`。

### 2.6 版本三连警告

磁盘上有三个不一致的版本号：源码 `_version.py`=**0.4.0**（权威）｜`rr_vfiqa.egg-info/PKG-INFO`=0.3.0｜`.venv` editable dist-info=0.1.0。任何用到版本号的工具（`test_package_contract.py` 除外）都可能读到旧值。

---

## 3. 仓库布局

### 3.1 顶层条目分类

| 类别 | 条目 | 说明 |
|---|---|---|
| 源码（提交） | `rr_vfiqa/` | Python 包本体 |
| 测试（提交） | `tests/` | pytest 套件（33 个 `test_*.py` + `conftest.py` + `real_corpus/`） |
| 工具（提交） | `tools/rpg_validation_generator/` | 程序化 RPG 验证数据生成器 |
| 脚本（提交） | `scripts/bench_1080p.py` | 一次性 1080p 基准 |
| 文档（提交） | `README.md`、`USERPLAN.md`、`PICPLAN.md`、`docs/` | 见 §7 |
| 配置（提交） | `pyproject.toml`、`setup.py`、`.github/`、`.gitignore`、`.claude/` | 构建/CI/权限 |
| 语料（gitignore，生成物） | `validation/`、`validation_draft/` | RPG 验证语料（见 §8.5） |
| 缓存（gitignore） | `cache/`、`_std/cache/`、`_b1080/cache/` | 光流特征缓存（见 §9） |
| 测试语料（gitignore） | `_std/`、`_b1080/` | standard / 1080p 测试视频 |
| 陈旧产物（可删） | `build/`、`rr_vfiqa.egg-info/`、`.venv/`、`.pytest_cache/`、`.serena/` | build 快照 / 旧元数据 / 死 venv |

> `.gitignore` 排除了 `validation/`、`validation_draft/`、`cache/`、`_std/`、`_b1080/`、`.venv/`、`build/`、`*.mp4`、`*.egg-info/` —— **语料和缓存都不进版本库**，CI 在测试时自行用生成器重建。

### 3.2 `rr_vfiqa/` 包地图（每个模块一行）

**顶层模块**

| 文件 | 职责 |
|---|---|
| `__init__.py` | 公共 API 门面：`evaluate_*`、`compare`、`EvaluationMode`、`EvalConfig`、`PRESETS`、`Report`、`__version__` |
| `_version.py` / `_build_info.py` | 版本（0.4.0）+ 构建提交元数据（供 `calibration/provenance.py` 做可复现哈希） |
| `config.py` | `EvaluationMode` 枚举、`Preset` 数据类、`PRESETS {fast, standard, audit, balanced}`、`EvalConfig` |
| `schema.py` | 三模式共享数据契约：`VideoMeta`、`Alignment`、`Window`、`WindowFeatures`、`MetricResult`、`Report` 等 + 数值工具 |
| `mode_router.py` | `inspect` 的自动安全路由 `route_mode()`、`SPEED_ALIASES`、`fail_closed_report()` |
| `multimode.py` | 公共调度 + **NR 与 FR 执行管线**（`evaluate_no_reference` / `evaluate_full_reference` / `compare`） |
| `pipeline.py` | **Endpoint-2x 执行管线** `evaluate_endpoint_reference()`、`evaluate_vfi()`（旧别名）、`compare_models()` |
| `cli.py` | argparse CLI：`evaluate` / `compare` / `inspect` / `inspect-batch`，`main()`、`_inspect_core()` |
| `benchmark.py` / `imutils.py` | 性能基准 CLI + 无依赖图像工具（`luma`、`phash64`、`mask_chamfer` …） |

**子包**

| 子包 | 职责 |
|---|---|
| `sampling/` | **Tier-1**：全片低分辨率扫描 `cheap_scan`、风险打分 `risk_score`、时间窗口选择 `window_selector`/`time_window_selector`、时间 NMS |
| `metrics/` | 每窗口指标族（标量 + 可选稠密 error map）：`no_reference`、`full_reference`、`parity_frequency`、`flow_composition_metric`、`cycle_reconstruction`、`global_technical_quality` 等 |
| `motion/` | 光流估计与几何：`flow_estimator`（Farneback/RAFT，内存感知 RAFT 微批）、`flow_composition`、`flow_geometry`、`global_camera_motion`、`occlusion` |
| `regions/` | 高风险区域诊断分支：UI 检测、卡片跟踪、人物分割、文字评估、细物体、转场、武器跟踪 |
| `diagnosis/` | **多证据诊断引擎**（近期重做）：`schema`（DiagnosticIssue/严重度带）、`rules`（8 条多证据规则 + issue 合并）、`cadence`（Cadence v2 硬门控 + 全片相位统计 + 独立/封顶惩罚模式） |
| `fusion/` | 分数融合：`score_schema`、`mode_score_schemas`（模式权重/契约串）、`feature_normalizer`、`feature_registry`、`monotonic_calibrator`（LightGBM） |
| `io/` | 视频解码 `video_reader`、端点/同帧对齐、颜色归一化 |
| `cache/` | **源码实现（不是数据）**：`source_cache.py`（`SourceCache`，按内容哈希/后端/宽度/契约哈希缓存光流特征）、`feature_store`、`cache_schema` |
| `report/` | 产物写出：`artifact_pipeline`（统一 heatmap→原帧→overlay→compare→keyframe，逐项失败隔离）、`html_report`、`json_report`、`badcase_exporter`、`heatmap_renderer` |
| `models/` | 可选学习后端：`flow_backend`（转发）、`tracker_backend`（KLT/CoTracker）、`segmentation_backend`、`depth_backend`（未实现）、`vqa_backend`（pyIQA） |
| `calibration/` | 启发式验证/标定：`detection_eval`、`model_validation`、`validate`（SROCC/PLCC）、`pseudo_gt`、`provenance`、`mode_validation` |
| `execution/` | 批量执行 `batch_runner.inspect_batch()` + index HTML |
| `testing/` | 合成视频与插值器生成器（`synth`、`interpolator`），供测试/标定 |
| `visualization/` | 空间诊断可视化（纯 OpenCV，无 matplotlib）：`maps`、`heatmaps`、`flow_overlay` |

---

## 4. 端到端数据流与 Tier 结构

```
cli.py
  ├─ evaluate / compare        → multimode.py  (NR + FR) 或 pipeline.py (endpoint)
  └─ inspect / inspect-batch   → mode_router.route_mode() → multimode.evaluate()
       │  自动路由：FPS 比 ≈2 → endpoint-2x；≈1 → full-reference；
       │            只有候选 → no-reference；无法安全满足契约 → fail-closed
       ▼
Tier-1  sampling/cheap_scan       全片逐帧，低分辨率 ~320-480px（场景切分/风险/相位统计）
       ▼
Tier-2  metrics/                  选中窗口 @ flow_width 960px：核心指标 + 稠密 error map
       ▼
Tier-3  _nr_fr_tier3 / endpoint   对最高风险窗口审计，tier3_max_width=960 封顶（VRAM 安全）
       ▼
diagnosis/  rules + cadence       每窗口标量 → DiagnosticIssue（8 规则 + Cadence v2 硬门控）
       ▼
fusion/                           特征 → 类别误差 → 子分数 → overall + confidence
       ▼
report/                           report.json / report.html / heatmaps / badcases
```

要点：

- **SourceCache 跨候选复用**：比较 N 个候选时只算一次源视频特征（按内容哈希缓存），成本 ≈ `T_source + N·T_candidate`。
- **inspect 自动路由 fail-closed**：证据不足时拒绝猜测并拒绝打分，生成 `failed` 报告。
- 三个 Tier 的资源占用不同，`--preset`/`--speed` 控制（见 §5）。

---

## 5. 模式、预设、速度（注意 `balanced` 撞名）

### 5.1 枚举与别名

```python
# config.py
EvaluationMode: no-reference | endpoint-2x | full-reference
PRESETS:        { fast, standard, audit, balanced }     # 真实预设

# mode_router.py
SPEED_ALIASES:  { fast: fast, balanced: standard, thorough: audit }   # --speed 的映射
```

### 5.2 ⚠️ `balanced` 撞名（易踩坑）

- `--speed balanced` → 解析为 **`standard`** 预设（`auto_audit=False`，**不含** Tier-3 自动审计）。
- 真正带自动审计的 `balanced` 预设（`auto_audit=True`）**只能**通过 `--preset balanced` 到达。
- CLI 还有一个怪癖：`inspect` 里**省略 `--speed`** 时，解析出的 `standard` 会被自动升级成 `balanced`（auto-audit）；而**显式 `--speed balanced`** 反而不会升级。

### 5.3 档位速查

| 期望 | 命令 |
|---|---|
| 快速 / 标准 / 深度审计 | `--speed fast` / `--speed balanced`(→standard) / `--speed thorough`(→audit) |
| 带 Tier-3 自动审计 | `--preset balanced` |

---

## 6. 契约、Schema 与规划文档

### 6.1 Schema 契约

三个模式的 Schema 版本串（见 §1 表格）由 `tests/test_docs_contract.py` 强制与 README / `docs/index.html` 同步。

**死代码警告**：`fusion/mode_score_schemas.py` 里 `METRIC_CONTRACTS` / `PRESET_CONTRACTS` / `ARTIFACT_CONTRACT` / `DIAGNOSIS_CONTRACT` **无任何引用**（PRESET_CONTRACTS 的键 `{fast, balanced, thorough}` 还与真实 `PRESETS` 键不匹配）。真实契约串是硬编码在 `multimode.py`（`nr-metrics-v3` / `fr-metrics-v3`）和 `pipeline.py`（`endpoint-metrics-v3` / `endpoint-reduced-reference-v3`）。两处来源可能漂移，改动契约时两处都要同步。

### 6.2 规划文档（活跃）

| 文档 | 内容 | 状态 |
|---|---|---|
| `USERPLAN.md` | P0/P1/P2 整改计划（Cadence v2、运动/UI 可靠性门控） | **工作区未提交版本**（中文一~十二编号结构） |
| `PICPLAN.md` | 程序化 RPG 验证数据生成计划（23 节） | 已落地为 `tools/rpg_validation_generator/` |
| `README.md` / `docs/index.html` / `docs/BENCHMARKS.md` | 三层文档系统 | README 是入口；`index.html` 是离线交互教程；`BENCHMARKS.md` 是性能证据 |

> 代码里大量 `USERPLAN §N` 引用（cli.py、pyproject、ci.yml、test_docs_contract.py…）对应的是**旧版已提交**的 USERPLAN 章节号，与当前未提交的中文编号版本**不一致**（见 §10）。

---

## 7. 测试与验证

### 7.1 pytest 配置

`pyproject.toml` → `[tool.pytest.ini_options]`：`testpaths=["tests"]`、`addopts="-q"`。无 markers/插件。CI 在 GitHub Actions（ubuntu-latest，Python 3.12）上跑。

### 7.2 测试文件与 USERPLAN 门控映射

**四个新增可靠性测试**（锁定 USERPLAN P0 门控）：

| 测试文件 | 锁定的门 |
|---|---|
| `test_cadence_regression.py` | §11 Cadence 梯度回归：True-60≤0.10、Duplicate-30→60≥0.70（三门全开）、Blend 低、战斗闪光不误报 |
| `test_flow_reliability.py` | §7 Flow Reliability Gate：闪光/粒子只升 `nr_effect_transient_*`、不判运动失败；`FLOW_VALID_FRACTION_MIN=0.25` |
| `test_phase_applicability.py` | §6 双相位适用性：`source_phase_likelihood≥0.7` 且 `phase_coherence≥0.6` 才进总分；权重重归一化 |
| `test_ui_reliability.py` | §8 UI/Text 可靠性门：静态固定坐标 HUD 可信；随机粒子掩码被拒 |

**三个改动测试**：`test_cadence_motion_gate.py`（Cadence v2 三门 + separate/cap15 惩罚模式）、`test_diagnosis.py`（多证据规则、support_spans→confirmed_affected_seconds、全局 vs 最差局部质量级、HTML 报告）、`test_multimode.py`（模式拆分、NR 60/120 时间归一化、FR fail-closed）。

其余按功能：`test_*_rpg_generator_*`（生成器）、`test_artifact_*`（产物失败隔离）、`test_docs_contract`、`test_motion_and_metrics`、`test_userplan_p0*` 等 33 个顶层文件 + `tests/real_corpus/test_metamorphic_regression.py`（15 个 RPG case 的定位+方向回归）。

### 7.3 CI 作业（`.github/workflows/ci.yml`）

| 作业 | 安装 | 跑什么 | 触发 |
|---|---|---|---|
| `unit` | `.[dev]` | 9 个纯函数/契约测试文件 | 每次 PR |
| `full-cpu` | `.[dev,fusion]` | `pytest tests/ --ignore=real_corpus`（**含四个新门控测试**） | 每次 PR |
| `smoke-e2e` | `.[dev,fusion]` | 5 个关键 E2E（含 media E2E、inspect、multimode） | 每次 PR |
| `metamorphic` | `.[dev,fusion]` | 全部 real_corpus + pipeline + multimode | main/定时/手动 |
| `gpu` | `.[dev,fusion,torch]` | 全套，`RR_VFIQA_TEST_FLOW=raft` | opt-in 自托管 GPU |

⚠️ 四个新增门控测试**只**在 `full-cpu`/`gpu` 跑，`unit` 作业不包含——本地只跑单元测试会漏掉它们。PR 前请在本地跑 `full-cpu` 级别。

### 7.4 环境变量

`RR_VFIQA_TEST_FLOW=farneback`（默认，CPU）| `=raft`（GPU 光流路径）。

### 7.5 `tests/conftest.py` 静默跳过警告

`conftest.py` 在 `try/except` 里导入 `rr_vfiqa.testing.synth`，失败时**静默**不定义合成视频 fixture。任何分支在 `_HAVE_RR_SYNTH` 上的测试会悄悄消失，不报 skip 汇总。如果测试数量明显偏少，先检查该导入是否成功。

### 7.6 语料语义

- `validation/`（1280×720/4s/120fps，正式）与 `validation_draft/`（640×360/2s，`--draft`）**都是程序化合成的**：`data_origin=procedural_synthetic_rpg`、`real_capture=false`、`production_gate=false`。`*_real` 后缀是遗留命名，**不代表真实采集**，不能用于生产门禁。
- 两者 gitignore，由 `python -m tools.rpg_validation_generator generate` 重建（生成器按 `case_specs.py` 单一真值表产出 15 个 case）。

---

## 8. 缓存与产物生命周期

### 8.1 缓存布局

| 位置 | 布局 | Schema | 谁写的 |
|---|---|---|---|
| `cache/`（根，默认 `cache_dir='./cache'`，相对仓库根） | `cache/<内容哈希>/flow_{backend}_{weights_hash}_w{width}_{contract_hash}/` | v2（含 cache_contract） | `rr_vfiqa/cache/source_cache.py` 在 pytest/评测时写入 |
| `_std/cache/`、`_b1080/cache/` | `cache/<哈希>/flow_w<width>/` | v1（无 cache_contract） | 旧版 SourceCache |

内容：光流/遮挡/相机模型的 `.npz` 特征（float16），**不是**模型权重，**不是**解码帧。缓存键 = 视频内容哈希 × 光流后端 × 宽度 × 契约哈希。

### 8.2 操作规则

- 旧布局（v1）条目在 `is_valid` 校验失败后会被**自动重算**。
- 强制全量重算：`rm -rf cache`。
- **跑测试会把仓库根 `cache/` 当作副作用写满**（4GB+），必要时清理；CI 每次是全新环境。
- 基准入口：`scripts/bench_1080p.py`（写 `_b1080/`，GPU RAFT 1080p）与 `python -m rr_vfiqa.benchmark`（默认 `./cache_bench`）。

---

## 9. 已知不一致与坑（索引）

1. **版本三连**：源码 0.4.0 / egg-info 0.3.0 / .venv 0.1.0（见 §2.6）。
2. **USERPLAN §N 引用失配**：代码里 `USERPLAN §N` 指旧提交版本；当前工作区版本是中文一~十二编号。
3. **README `-- speed` 排版错误**（`--speed` 中间多了空格）+ 「balanced 默认含 tier-3 审计」不成立（tier-3 需 `--preset balanced` + audit extra）。
4. **`--speed balanced` 撞名**：实际是 `standard` 预设，不含自动审计（见 §5.2）。
5. **`schema.py::Window.start_time` 是 NaN stub**：要用 `sampling.window_selector.window_times` 取真实时间。
6. **`pipeline.py::_classify_window` 硬编码 `/120.0`** 起始时间假设（注释说调用方会用真实 PTS 覆盖，读早了会误标）。
7. **`docs/calibration/synthetic_detection_12class.json` 是历史 0.1.0 产物**（numpy 2.5.1 / opencv 5.0.0 等），别当现役规格。
8. **`fusion/mode_score_schemas.py` 契约常量是死代码**（见 §6.1）。
9. **语料不可由 provenance 单独复现**：生成时 working-tree 是脏的，`validation*` 又 gitignore。
10. **ds-torch 缺 pyiqa/cotracker**：vqa、audit 功能当前不可跑（见 §2.4）。

---

## 10. 贡献与上手

**新代码去哪：**

| 想做什么 | 改哪里 |
|---|---|
| 加一个指标 | `metrics/`：`compute_window` + `compute_window_maps` |
| 加一条诊断证据 | `diagnosis/rules.py` + `diagnosis/cadence.py` 硬门控 |
| 加一个子分数 / 权重 | `fusion/mode_score_schemas.py` |
| 加一个产物/报告 | `report/artifact_pipeline.py`（逐项失败隔离模式） |
| 改 CLI 契约串 | `multimode.py` / `pipeline.py` 硬编码处 + `fusion/mode_score_schemas.py` 死代码处 **两处同步** |

**提交前：**

1. 改文档后跑 `python -m pytest tests/test_docs_contract.py`（README 与 `docs/index.html` 与生产代码同步）。
2. 本地跑 `python -m pytest`（CPU/Farneback）；改 cadence/flow/phase/UI 门控的跑四个新增可靠性测试 + `test_cadence_motion_gate.py`。CI 的 `unit` 作业不含这些，别只靠它。
3. 本地 `tests/real_corpus/` 有约 10 个环境性失败（语料生成 + 系统 python 的 farneback）属已知现象，不以它们为合并阻断；CI 的 `metamorphic` 作业在干净环境跑全量。

**当前活跃开发面**：`diagnosis/`（cadence v2、rules、schema）与 `metrics/`（`no_reference.py`、`parity_frequency.py`）——对应未提交的 USERPLAN P0/P1 整改。
