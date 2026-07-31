# 结论

最新提交 `5618ac80` 对上一轮问题进行了有针对性的整改。多 Issue 文件覆盖、Error Map 归一化、显式模式传递、preferred map 选择、动态版本和特征契约等实现已经落地。

**但测试整改仍未完全形成可靠的质量门禁。** 当前状态更准确地说是：

| 项目          |        评价 |
| ----------- | --------: |
| 被测实现整改      | **约 90%** |
| 基础单元测试      | **约 70%** |
| 三模式端到端测试    | **约 60%** |
| 变形测试有效性     | **约 45%** |
| 媒体证据测试      | **约 55%** |
| 当前测试能否证明可发布 |    **不能** |

Review 结论建议为：

```text
实现可进入内部 Beta
测试体系仍需 Request Changes
```

最新提交没有可见的 GitHub Actions workflow run 或状态检查，因此目前也不能声称测试已经实际通过。 我的执行环境仍无法解析 GitHub 域名，无法独立 clone 后运行 pytest，以下结论来自最新代码和测试定义的静态审查。

---

# 一、这次测试整改已经完成的部分

## 1. Endpoint reference 获取逻辑已经修复

测试会从 manifest 中读取 Endpoint 和 FR 的 `source` 作为 reference，不再默认把 Endpoint case 当作 NR 运行。

## 2. Metric Direction 已把错误方向纳入分母

现在统计：

```text
correct
wrong
inconclusive
```

方向错误不再被遗漏，正确率分母改为 `correct + wrong`。

这是必要的修正。

## 3. 默认媒体路径已有 E2E 测试

测试不再全部使用 `--no-clips`，已经开始验证：

* JSON 和 HTML；
  -热力图路径；
  -Overlay/Compare 路径；
  -PyAV 能否解码导出视频。

## 4. 多 Issue 覆盖问题已有回归测试

新增测试会寻找三个以上问题，检查多个 Overlay 文件存在且内容 hash 不同。

实现侧也已经改成一次批量调用 exporter，不再逐个调用并反复生成 `issue_000`。

## 5. 静态视频 Cadence 已有真实视频测试

不再只是手工构造 scalar，而是编码一段静态视频并运行完整 `inspect`，验证 cadence risk 接近零。

---

# 二、测试数据存在一个严重基础问题

## RPG 视频只有 2 秒，但缺陷区间最长到 3.1 秒

测试 fixture 将所有 RPG 视频生成为：

```python
duration_seconds=2.0
```

但 case 定义中的缺陷区间包括：

```text
1.20–2.80 秒
1.00–3.10 秒
1.30–2.70 秒
1.00–3.00 秒
```

这意味着许多缺陷：

-只注入了一部分；
-结束阶段被裁掉；
-没有缺陷后的恢复区间；
-实际视频缺陷时长与 manifest 标注不一致；
-定位测试仍用原始的 2.8 或 3.1 秒作为 GT。

这很可能正是 Endpoint 定位能力低、测试不得不把门槛降到零的原因之一。

## 修复方案

测试视频时长不能固定为 2 秒，应从 case spec 动态推导：

```python
duration_seconds = max(
    case.end_time for case in CASES
) + 0.5
```

当前数据至少应生成约 3.6 秒，建议直接使用 4 秒。

更严格的生成器还应断言：

```python
assert defect.start_time < video_duration
assert defect.end_time <= video_duration
assert defect.end_time - defect.start_time >= minimum_defect_duration
```

---

# 三、Endpoint 定位测试目前等价于没有测试

当前 Endpoint localization floor 是：

```python
floor = 0.0
```

所以 Endpoint 五个 case 全部定位失败，测试仍然通过。

Endpoint 是三种模式中最成熟、参考条件最强的模式，它不应该拥有最低的测试要求。

## 推荐门槛

完成 4 秒、640×360 测试数据修复后，最低应设为：

| 阶段      | Endpoint 定位率 |
| ------- | -----------: |
| 当前整改验收  |         ≥40% |
| 内部 Beta |         ≥65% |
| 稳定版本    |         ≥80% |

不能通过把门槛设置为零来适配检测器，而应提高 fixture 的缺陷强度、分辨率和时长。

---

# 四、模式断言写了，但没有真正使用

`_inspect()` 已经支持：

```python
expected_mode
```

并可断言报告模式。

但目前 localization、score 和 metric-direction 调用都没有传入 `expected_mode`。例如 localization 只是：

```python
_inspect(..., mode=mode)
```

`mode` 参数本身也没有加入 CLI 命令。

因此即使：

* Endpoint 错误路由成 NR；
  -FR 错误路由成 Endpoint；

测试仍可能继续读取分数和特征，并将结果记为 inconclusive 或跳过。

## 修复方案

所有 case 评测必须写成：

```python
rep = _inspect(
    case["candidate"],
    _reference_for_case(case),
    output_dir,
    mode=mode,
    expected_mode=mode,
)
```

并在 `_inspect()` 中加入：

```python
rc = main(cmd)

assert rjson.exists(), "inspect did not write report.json"

rep = json.loads(rjson.read_text())
expected = _MODE_ALIAS_TO_META[expected_mode]
assert rep["meta"]["mode"] == expected

if rep["meta"]["status"] == "failed":
    assert rc == 1
else:
    assert rc == 0
```

目前 `_inspect()` 忽略 CLI 返回码，报告不存在时只返回 `{}`，这会把真正的执行失败转换为后续的 skip 或 inconclusive。

---

# 五、媒体 E2E 仍可能空通过

当前媒体测试只在 `issues` 非空时检查卡片和媒体路径。若选中的第一个 case 没有触发任何 Issue：

```text
issues = []
map 检查循环不执行
clip 检查循环不执行
视频解码检查不执行
```

测试仍然通过。

它还只严格断言 NR 模式；Endpoint 和 FR 的实际路由仍未验证。

## 修复方案

不要使用“每种模式的第一个 case”，而应指定三个保证触发问题的强缺陷 case：

```text
NR：长时间 freeze / cadence collapse
Endpoint：大范围 generated_freeze_copy
FR：大范围 global_blur 或 spatial shift
```

测试必须要求：

```python
assert rep["meta"]["mode"] == expected_mode
assert rep["meta"]["status"] != "failed"

issues = rep["meta"]["diagnostics"]["issues"]
assert issues, "strong defect produced no diagnostic issue"

issue = issues[0]
assert issue["maps"]
assert "overlay" in issue["clip_paths"]
assert "compare" in issue["clip_paths"]
assert issue["thumbnail"]
```

然后逐个验证：

-文件存在；
-非零大小；
-可以解码；
-帧数至少为 2；
-HTML 包含相同相对路径。

---

# 六、多 Issue 测试仍可整体跳过

当前测试会遍历 NR case，寻找三个以上 Issue；如果没有找到，则：

```python
pytest.skip(...)
```

因此文件覆盖 bug 重新出现时，只要检测器没有产生三个 Issue，这个专门的回归测试仍可能不执行。

它也只检查 Overlay，没有检查：

* Compare；
  -Keyframe；
  -Heatmap；
  -报告中的路径是否唯一。

## 推荐改成确定性单元测试

不要依赖诊断器生成 Issue。直接构造三个 Issue：

```python
issues = [
    make_issue(0.2, 0.4, "freeze"),
    make_issue(0.6, 0.8, "blur"),
    make_issue(1.0, 1.2, "ghost"),
]
```

使用一个固定的小视频直接调用 Artifact Pipeline，然后断言：

```python
expected = {
    "issue_000_overlay.mp4",
    "issue_001_overlay.mp4",
    "issue_002_overlay.mp4",
    "issue_000_compare.mp4",
    "issue_001_compare.mp4",
    "issue_002_compare.mp4",
}

assert expected <= {p.name for p in badcases.iterdir()}
assert len(set(issue["clip_paths"]["overlay"] for issue in issues)) == 3
assert len(set(issue["clip_paths"]["compare"] for issue in issues)) == 3
```

该测试不得 skip。

---

# 七、静态视频测试也能空通过

当前逻辑是：

```python
if os.path.exists(rjson):
    assert cadence_risk < 0.1
```

如果 `report.json` 根本没有生成，测试不会失败。

应改为：

```python
assert os.path.exists(rjson), "static inspect did not produce report.json"

rep = ...
assert rep["meta"]["mode"] == "no-reference"
assert rep["meta"]["status"] != "failed"
assert rep["meta"]["cadence"]["cadence_risk"] < 0.02
assert rep["scores"]["cadence_integrity"] >= 98.0
```

---

# 八、当前变形测试门槛仍然过低

现有门槛大致为：

| 项目              | 当前门槛 |
| --------------- | ---: |
| NR 定位率          |  40% |
| Endpoint 定位率    |   0% |
| FR 定位率          |  20% |
| 总分方向            |  50% |
| 指标方向            |  30% |
| 允许 inconclusive |  70% |

这样的门槛只能作为 smoke test，不能作为回归门禁。

## 内部 Beta 建议门槛

| 项目                   | 建议门槛 |
| -------------------- | ---: |
| NR 定位率               | ≥60% |
| Endpoint 定位率         | ≥60% |
| FR 定位率               | ≥70% |
| 总分下降方向               | ≥75% |
| 目标指标方向               | ≥70% |
| Inconclusive 比例      | ≤30% |
| 每个模式 conclusive case |   ≥3 |
| Clean severe 误报率     | ≤10% |

测试不应在 `valid == 0` 或 `conclusive == 0` 时直接 skip。应要求最少有效样本数：

```python
assert valid >= 3
assert conclusive >= 3
```

---

# 九、新实现缺少对应的直接单元测试

最新提交新增或修改了以下关键机制：

* Error Map P99.5 normalization；
  -Overlay resize；
  -显式 mode；
  -preferred map 选择；
  -Artifact Status；
  -动态 package version；
  -Feature Registry resolution contract。

实现本身已经存在，例如 Overlay 会 resize 并归一化真实 map。

但目前测试主要通过完整 RPG pipeline 间接覆盖，问题定位困难。建议增加以下独立测试文件：

```text
tests/test_artifact_pipeline.py
tests/test_error_map_contract.py
tests/test_diagnosis_map_selection.py
tests/test_package_contract.py
```

## 必须增加的单元测试

### Error Map normalization

```python
def test_normalize_map_p995_keeps_dynamic_range():
    x = np.linspace(0, 255, 10000).reshape(100, 100)
    y = _normalize_map(x)
    assert y.min() == 0
    assert 0.9 < np.percentile(y, 95) <= 1.0
```

### Preferred Map

构造两个热区位于不同位置的 map：

```text
composition map：左上热
duplicate map：右下热
```

对 `duplicate_freeze` 诊断，box 必须位于右下，并且 `issue.maps` 必须是 duplicate map。

### Artifact failure isolation

Monkeypatch Overlay exporter 抛异常，要求：

```text
overall score 保持有效
meta.status 不变
artifact_status.overlay_clips.status = degraded
report.json 中能读到该状态
```

### Package version

```python
assert importlib.metadata.version("rr-vfiqa") == rr_vfiqa.__version__
```

### Feature Contract

```python
assert fr_edge_chamfer.units == "frame-diagonal-ratio"
assert fr_edge_chamfer.resolution_invariant
```

---

# 十、Artifact Status 仍有一个测试应当捕获的实现缺口

当前在 atomic write 前将 `artifact_status` 写入 meta；如果 atomic write 自身失败，代码随后把：

```text
report_write.status = degraded
```

写进内存，但不会再次落盘。

因此磁盘中的基础报告仍可能显示 `report_write=ok`，尽管最终写入失败。

需要设计：

```python
monkeypatch.setattr(os, "replace", raise_error)
```

然后验证磁盘报告如何表达写入失败。比较可靠的策略是：

1. 先写基础报告；
2. 尝试 atomic final write；
3. 失败后更新 `report_write=degraded`；
4. 使用普通覆盖方式尽力更新基础 JSON；
5. 保证至少 JSON 中包含失败状态。

---

# 十一、CI 需要拆分，否则测试成本和故障定位都会失控

当前一共有 15 个 RPG case。

现有测试大约会执行：

```text
Localization：15 次 inspect
Score：30 次 inspect
Metric Direction：30 次 inspect
Media E2E：3 次 inspect
Multi Issue 搜索：最多 5 次 inspect
Static：1 次 inspect
```

合计约 **84 次完整评测**，而且大量 candidate/oracle 被重复计算。

建议拆成：

```text
unit:
  普通函数、Schema、map、artifact 测试

smoke-e2e:
  每种模式 1 个强缺陷
  每个 PR 执行

metamorphic:
  全部 15 case
  nightly / 手动 / main 合并前执行

gpu:
  RAFT 路径
  自托管 GPU
```

并增加 session 级评测缓存：

```python
inspection_cache[(case_id, role, mode, clips)] = report
```

让 localization、score 和 metric-direction 共用同一份 no-clips 评测结果，预计可将约 75 次 no-clips 评测降低到约 30 次。

---

# 最终评价

这次整改已经解决了大量实现问题，尤其是多 Issue 导出、Error Map normalization 和 preferred map 选择。但**测试本身仍存在大量可跳过、空通过和错误数据区间问题**。

当前最优先的四项测试整改是：

1. 将 RPG 时长从 2 秒改为至少 4 秒；
2. 所有评测强制断言实际模式和 CLI 返回码；
3. Endpoint 定位门槛从 0 提升到至少 40%～60%；
4. 将媒体、多 Issue、preferred map 和 failure isolation 改为确定性、不可 skip 的单元/E2E 测试。

完成这些后，测试体系才足以支持：

```text
APPROVE FOR INTERNAL BETA
```

当前实现已经接近这一阶段，但现有绿灯仍不足以证明三模式及媒体证据链可靠。
