# 模型结构到预取实现

## 先完成结构表

对每个候选记录以下内容。配置的逻辑维度不能代替运行时分片和存储布局。

| 对象 | 必需信息 | 证明来源 |
|---|---|---|
| 模型/阶段 | model type、层数/层类型、decode/prefill/vision/MTP 边界 | config、forward、调度/图 marker |
| 消费者 | 实际融合模块名、源码位置、算子、输入及输出 shape | forward、kernel CSV、图节点 |
| 权重 | post-loader Parameter、rank-local shape/dtype/format/stride、有效 storage 范围、reload 生命周期 | loader/后处理与设备审计 |
| 通信 | TP/EP/DP group、AllReduce 等实际调用、物理 stream、同步语义 | 通信封装、trace/communication.json |
| 路由 | topk、expert map、counts/cumsum/KV-list 语义、路由可用时刻 | router/dispatch 实现与原始路由 |
| 图 | eager/capture/replay、静态输入地址、动态值更新、fake/meta 与 side effects | 编译路径和真实图验证 |

通过源码顺序映射 kernel。若没有 task/layer marker，给出多个可能映射；不要只按耗时聚类就赋予层名。无消费者边界就不能给“consumer wait 已消除”的结论。

## 模型族适配

| 模型族 | 优先检查的目标 | AllReduce 窗口候选 | 关键限制 |
|---|---|---|---|
| Dense Transformer | QKV、gate/up、down、输出投影 | 当前层输出归约期间的后续固定投影 | 融合 QKV 与分离 q/k/v 不同；Row/ColumnParallel 的归约位置不同 |
| MoE | routed GMM1/GMM2、共享专家、router、小型固定投影 | 先选地址已知的后续固定权重；专家预取需要路由已就绪 | 目标专家 ID 和 token counts 必须属于目标层；EP 改变本地专家及映射 |
| Hybrid attention/GDN | 不同层型的 in-proj/out-proj、随后 MoE | 按实际层序找到后续消费者 | 不假定全部是 self-attention；静态卷积重排是独立的加载期优化 |
| MLA/融合 attention | 实际 q/kv down/up 或融合 prolog 权重 | 上游通信与该融合消费者之间 | 原始 checkpoint 权重可能已拼接、打包或转换 |
| W8/W4 等量化 | 量化投影的真实 packed storage 与必要 scales | 消费者前的独立通信 | 使用分配后的实际字节/offset；不能按 BF16 大小，也不能将 packed weight 任意 cast/view 来触达 |

这些是查源码的入口，不代表任意具体版本均有这些模块或窗口。TP/EP/PP、sequence parallel、FlashComm/MC2、LoRA、speculative decoding、重载/卸载都会改变 DAG 与 fallback。

## 合法的 AllReduce 并发

```text
producer 完成 ─┬─ 主/通信流：AllReduce ── 后续必要操作 ── consumer
              └─ 预取流：目标权重预取 ────────────────┘ join
```

预取流只等待启动前必要的 producer；在入队 AllReduce 之后再对主流调用 wait_stream，可能把通信也纳入依赖而串行。以实际 NPU stream/event 和 capture 语义验证，不能从 Python 调用先后猜测。消费者首次使用前等待该目标的预取完成；不要等待其他层无关目标。是否能把 join 放到通信之后更晚的首个消费者处，由真实依赖决定。

MoE 常见实际顺序是 `router → dispatch → GMM1 → activation → GMM2 → combine → AllReduce`。若目标是该层 W2，末尾 AllReduce 已经太晚；若换成下一层 W2，下一层路由通常尚未知。两者不能合并成“真实路由 W2 与末尾 AllReduce 重叠”。应拒绝该精确路由候选，或明确转为有命中率证据的预测实验。Dense 的下层固定 QKV 可以是另一候选，但需单独证明消费者访存瓶颈。

## 预算与收益估计

未压缩 ND 权重逻辑字节可用 `prod(shape) * element_size` 交叉校验；真实边界仍由分片后 storage、offset、format 和 API 限制决定。对 NZ/packed/非连续 view，先确认 API 支持其实际布局，不在 decode 临时做 contiguous/转置来迁就预取。

预算同时受以下因素约束：目标合法字节数、可用 cache、有效通信窗口、实测可用搬运速率和其他流竞争。选项目已有的小预算作起点；一次只改一个预算或插入点。专家预算须写明“每 rank/每层/每次调用”，并计算覆盖多少个专家或专家分片。按专家编号截取会有偏置，热度排序和消费顺序不一定一致。

静态上界以消费者**暴露的**加载为基础：`potential <= min(exposed_load, useful_overlap) - launch - tail - contention`。MTE2 流水活跃时间不等于 exposed_load。缺少依赖等待、cache/带宽或消融证据时只给定性假设；不拿此公式预测保证的 TPOT 百分比。

## 代码改动清单

复用目标框架已有的控制器/配置/通信封装，最小实现通常只需要：配置贯通、加载后绑定、通信前提交、首消费者 join、fallback 及相关测试。不要为预取重写模型 forward。

- 配置：沿用配置对象或集中 env 注册；实验关闭路径必须保持普通实现。
- 绑定：在分片、静态重排和 quant 后处理后，首次图捕获前绑定实际权重；记录身份、地址、offset、shape、format。重载时重新检查和重捕获，避免旧图读旧存储。
- 调度：复用侧流和事件；捕获前创建需要的持久对象，维护每设备在途目标。空 batch/混合 prefill/不支持路径正常回退。
- 图与 ABI：检查 schema、fake/meta/副作用、host/device 结构偏移、注册/动态库、编译目标。两个结构总字节相同不代表字段偏移一致。不要把诊断期字节搬移适配器当生产 ABI。
- 路由：复用设备输出；若必须编码，保持所有 expert/count/row 顺序语义，计入编码+等待+GMM 的完整成本。参数存在不等于设备 kernel 实际消费了该 dependency。
- 精度：预取原则上只读权重；输出差异需通过 A/A、固定路由/调度及消费者张量诊断定位。用户允许 token 不同时，以业务规则验收，不额外恢复逐 token 相同约束。

## 测试与部署

先证明 off/unsupported fallback、预算边界和 reload；再用同输入/同逻辑权重/同路由测 eager、capture、动态路由 replay。ND→NZ、原生→自定义核、编码优化、预取分别消融，不把布局收益算成预取收益。

设备单点实验包含 off、相同启动/路由开销但不读权重的 control、真实 prefetch。测两 rank 的目标、前序通信、tail、consumer wait、模块跨度和异常；空操作 control 会有不同资源占用，不能简单做数值相减当因果贡献。

服务用交替配对会话，输入/模型/硬件/并行/图/并发固定，缓存根目录和热态协议可复现。profile 和无 profiler 服务测量分开；至少报告每对会话而非仅合并平均值。重复数和阈值由用户/实验设计决定。通过性能仍需业务精度与异常门；记录源码/二进制 hash、命令、数据位置及最后启用状态。
