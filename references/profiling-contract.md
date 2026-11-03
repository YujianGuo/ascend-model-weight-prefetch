# 分析脚本数据合约

`scripts/analyze_prefetch.py` 使用 Python 3.10+ 标准库。本工具分析证据并检查候选必要条件，源码解读和模型实现由 skill 执行。

## 文件与范围

每个输入目录可包含 `kernel_details.csv`、`trace_view.json` 或 `trace_view.json.gz`。至少一个存在。脚本不递归挑选 rank/采集，必须给出精确目录；不同 rank 分别运行并一起审查。缺文件报告 unavailable，合法空 trace 才是测得零事件。非法/截断 JSON 失败退出。DB 和 `msprof_*.json` 不是当前支持的替代格式，需要用匹配版本的离线 profiler 导出。

输出 JSON 和可选 Markdown；已有输出和任何输入文件不会被覆盖。原始采集只读。输出带输入 SHA256，避免把旧 profile 当新运行。

## Manifest

参考 `manifest.example.json`。它是示例结构，未填写的证据字段保持 null。

- `model`：人可读模型信息和源码出处，不用模型名推断层结构。
- `require_allreduce`：是否将 AllReduce 作为预取窗口的硬约束。示例默认 true；应按目标任务要求设置，并验证通信类型及源码顺序。
- `runs.baseline/candidate`：模型/权重 revision、TP/EP/PP、graph、量化、workload_hash、source_diff_evidence。前后源码可以不同；`source_diff_evidence` 填同一份已审核窄 diff 的标识，表示比较已排除无关改动。缺项/不等只形成证据问题，不签因果结论。
- `components[]`：name、name_regex（匹配 kernel Name）、shape_regex（匹配去引号空白后的 Input Shapes）、stage、mapping_evidence。每个 CSV 行最多匹配一个 component，否则失败。只分桶被映射的行，保留未映射数量。用户必须从源码/marker证实 stage；脚本不会由 M=1/32/256 自动推断 decode，也不自动分层。
- `trace`：prefetch_regex、communication_regex、join_regex，可选 prefetch_pid/communication_pid 与 prefetch_stream/communication_stream。选一组真实设备事件；CPU `cpu_op/enqueue/dequeue` 不计重叠。HCCL 常有高层/内核重复事件，不同时选两个别名来扩大时间。`mapping_evidence` 写预取开关/源码和 event 对应证据；CMO 本身不能自动归因目标权重。
- `runs.<label>.trace_windows_us`：可选绝对微秒起止窗口数组，例如 `[["1791775424750000.000", "1791775424760000.000"]]`。以字符串保留精度。窗口来自已验证的 decode replay 边界；无窗口时明确标记整个 capture，禁止称为纯 decode。支持多个不相交窗口。
- `candidates[]`：name、routing（none/exact/speculative）、weight.storage_bytes/binding_evidence、budget_bytes、alignment_bytes、window.kind/consumer_after_window/route_known_before_window/source_evidence、memory_evidence、start_site、join_site。预测路由还需要 route_hit_rate_evidence。shape/dtype/format 可作为人工审阅补充，实际字节以 post-loader 审计为准。

`ready_for_single_point_experiment` 只表示 manifest 声明的必要证据齐备，不表示脚本已经验证源码，也不表示安全/性能通过。AllReduce 在消费者之后、精确路由未就绪、越界或不对齐预算会标记 ineligible。未知证据标记 needs_evidence。

## Kernel 统计

比较签名为 `(component, stage, operator Type, input shape/dtype/format, output shape/dtype/format)`。缺少 Type 时保留完整 Name。变化的 kernel 名单另外列出。signature 不同的行绝不生成加速百分比；相同 signature 仍需相同 expert IDs/counts、cache、调度和基准 workload 才可归因。

Duration、Wait Time、aic_mte2_time、aiv_mte2_time 分别统计有效样本数、mean、p50、p95。p50 使用中位数；p95 在 `(n-1)*0.95` 处线性插值。N/A/缺字段是 unknown，不填零；负值/非有限数字拒绝。MTE2 不相加为额外关键路径时间，Wait Time 含框架/流水定义，也不自动称为权重等待。

脚本对比的是单个 kernel/event 分布。完整 replay 的跨度、跨 rank 等待、真正 consumer stall 和客户端 TPOT 需要源码边界与服务测量。某段 device task 数改变时，同时审查 graph capture 次数、batch、路由及 tiling，不能比较原始总和。

## Trace 统计

支持顶层数组或 `{"traceEvents":[...]}`，包括 gzip。流式解析逐个事件，内存不随全部 CPU 事件增长；仅保留选中设备区间。微秒数字/字符串先转整数纳秒，避免约 10^15 微秒绝对时间戳直接转 float 后丢失小数。

只计算 `ph=X` 的完整事件。分别合并预取区间 P 与通信区间 C，输出：

- 预取事件区间数、累计时长及并集时长；如显式时间窗口将一个事件切成多段，区间数是切片数。
- 通信并集时长、`length(P ∩ C)`、`length(P ∩ C)/length(P)`。
- 与通信至少有一次时间交叠的预取区间数，及 join 的 CPU span 分布。

重复/嵌套通信不会重复计时，重复预取不会抬高并集比例。prefetch_sum_us 与 prefetch_union_us 口径不同。join CPU span 不等于设备等待，也不能直接从相交率推断 cache 命中或收益。未指定时域/stream/目标归属的结果只能称 capture 级时间重叠。

## 示例命令

```bash
python scripts/analyze_prefetch.py --baseline /profiles/base/ASCEND_PROFILER_OUTPUT \
  --manifest /work/model.json --output /work/baseline-observations.json

python scripts/analyze_prefetch.py --baseline /profiles/base/ASCEND_PROFILER_OUTPUT \
  --candidate /profiles/candidate/ASCEND_PROFILER_OUTPUT --manifest /work/model.json \
  --output /work/comparison-01.json --markdown /work/comparison-01.md
```

输出不会自动读取邻近目录的业务 benchmark，避免误配旧结果。服务测量独立核对输入 ID/内容、生成策略、输出长度、并发和 profiler 开关，按用户标准评估业务正确率/EOS/异常。报告中同时保留 kernel、trace 和服务三个口径及各自数据来源。
