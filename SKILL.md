---
name: ascend-model-weight-prefetch
description: 根据实际昇腾模型源码、运行时权重分片及 profiling，分析并实现权重预取；覆盖 Dense、MoE、混合注意力与量化模型的通信重叠、图模式适配和服务配对验收。用于跨模型预取开发及已有候选诊断。
---

# 按模型结构与 profiling 开发权重预取

交付真实源码改动、可重放的分析、单点实验和服务验证。先识别消费者及合法依赖窗口，再决定是否值得写预取代码。已有目标和开发授权时继续执行，不重复要求方案确认；只要求分析时保留分析边界。

## 输入与证据

取得目标容器/源码、模型配置、实际 TP/EP/PP、dtype/量化、torch-npu/CANN/硬件版本、图模式、目标阶段和 profiling 路径。记录实际 consumer、post-loader 权重 shape/stride/format/地址与字节数。已有连接和脚本可复用；凭据不进入 skill、manifest 或报告。

- 先读 [模型适配与实现](references/model-adaptation.md)，按实际源码重建 producer→通信→consumer DAG；Dense/MoE/混合模型的选择规则在同一文件。
- 分析采集数据时读 [profiling 合约](references/profiling-contract.md)，使用随附脚本；它不需要 torch、NPU 或第三方包。
- 新模型结论必须来自对应源码、权重布局和 profiling；不要把其他模型的模块映射或历史数值迁移为当前收益。

缺少新目标时可以完成分析工具和方案，但明确区分工具验证与模型部署验收。

## 工作流程

1. **源码与采样对应。** 核对模型配置和权重分片、forward/loader、融合投影、router/dispatch/combine、通信封装、图捕获/重放路径。用源码/图 marker 校验层序与阶段，不以算子名称或小 M 单独定义 decode。将对应关系写成 manifest；未覆盖和多义映射显式列出。
2. **确认瓶颈。** 按阶段、模块、shape、dtype、format、rank、真实路由和缓存状态比较 Duration、消费者等待、MTE2 及关键路径。MTE2 比例高是候选线索；用带宽/cache 计数、固定权重/路由消融和设备事件继续确认。不能按 MatMul/通信/Norm 的类别直接断言是否 memory-bound。
3. **选择通信窗口。** 用户要求与 AllReduce 同时执行时，此要求作为硬约束。预取侧流等待通信前必要的 producer，就绪后与 AllReduce 并行；消费者前汇合。若把侧流依赖放在 AllReduce 之后，窗口已经丢失。画出源码实际先后关系并验证 graph replay；CPU API 同时入队不足以证明设备重叠。
4. **检查目标是否可知。** 同层 W2 若先消费、后 AllReduce，不能用该 AllReduce 隐藏该次 W2 的加载。跨层 Dense/QKV 权重地址通常可提前确定，但也需检查动态权重。下一层 MoE 路由通常在上一层 AllReduce 后才产生，精确专家预取因此不可行；预测性预取作为独立假设，先测命中率及额外带宽。当前层路由不能冒充下一层路由。
5. **设计最小候选。** 绑定加载/分片/格式转换后的实际存储，预算满足 API 对齐、offset 和 storage 边界。依据每 rank 权重、有效窗口、实测搬运速率、cache 容量/驱逐和其他在途目标定预算，不默认搬半个或整个专家库。复用项目配置与预取流，单点起步；保留 capability fallback，实验默认关闭，除非用户明确另有要求。
6. **实现完整调用链。** 列清配置→初始化→post-loader 绑定→launch→consumer join→fallback/reload，以及 custom-op schema、fake/meta、ABI、注册和构建调用者。使用当前安装版本的预取 API；先查本地实现/文档，版本不确定再核对官方资料。保持每设备在途对象和事件生命周期受控；decode 不新增路由 `.cpu()`/`.item()`。权重替换后重绑且按框架要求重新捕获图。
7. **验证并作决定。** 测预算/存储/错误分支、eager 与图模式、动态路由 replay、双 rank；然后固定输入/权重/路由比较 A0、空操作 control、真实预取。检测任务/CMO、AllReduce 时间交集、通信变慢、预取拖尾、消费者等待及模块跨度。单点无有用改善时停止扩预算，记录负结果。kernel 样本数不等于独立服务会话数。
8. **服务验收。** 排除加载和编译，等健康就绪；无 profiler 做交替、重复配对，固定输入、并发、TP/EP、图和生成策略，报告 TPOT mean/p50/p95、吞吐、错误及每会话差值。采样抖动/输出长度变动需解释。遵守用户的业务正确率、EOS/截断/重复/乱码要求；原始 token 可用于诊断，是否逐 token 一致由用户验收约定决定。未测的门不签通过。

## 分析脚本

从 `references/manifest.example.json` 复制并填入源码审核后的映射；示例的证据字段留空，不能当作已核验配置。可以只分析 baseline，也可以比较两个目录。

```bash
python scripts/analyze_prefetch.py --baseline /profiles/base/ASCEND_PROFILER_OUTPUT \
  --candidate /profiles/on/ASCEND_PROFILER_OUTPUT --manifest /work/model-manifest.json \
  --output /work/result-01.json --markdown /work/result-01.md
python -m unittest discover -s tests -v
```

脚本支持 Chrome trace 数组、`traceEvents` 对象及 gzip，流式解析；整数纳秒计算保留大时间戳精度。区间取并集后再求交，防止重复/嵌套 HCCL 事件抬高比例。无 trace 与测到 0 不同。它只比较完整 shape/dtype/format 签名，且不会把 CMO 自动归因到某个权重、把重叠率换算 TPOT 或写模型源码。

## 交付与结论

交付结构/阶段映射、候选表（目标存储、预算、route 可用时刻、launch/join 源码点、通信窗口及预期副作用）、窄代码 diff、测试/采集脚本、原始数据路径/hash、分析报告和回退状态。静态估计只给上界及假设：可隐藏的消费者暴露加载减去 launch、尾部等待和争用；没有实测消费者暴露加载就不给数值 TPOT 承诺。

分别判定：预取路径触发、目标归因、合法依赖、设备重叠、消费者加速、服务收益、业务正确性。任一项未知均保留未知，不让前一项替代后一项。没有适合窗口或候选未过门时，明确给出不启用的结论及证据。

