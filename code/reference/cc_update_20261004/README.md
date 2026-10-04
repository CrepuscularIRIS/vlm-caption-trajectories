# cc 补审代码阅读快照

本目录补充前次发布时尚未验证的 cc 路由实现。[来源与变换哈希](../../../data/cc_code_provenance_20261004.json)。统计仍以 [01:22 完成快照](../../../data/cc_update_20261004.json) 为准，源码捕获时点单列。

- `funnel_audit.py`：实际审核入口，可选原官方 CLI 或新增 cc 包装；保持结构审查、前缀盲读、隔离比较与分歧处理。
- `cc_proxy_reviewer.py`：cc 适配器；仅 Read 指定图像，显式 medium effort，独立回执/预算键，按回执验证模型并接受同型号 `[1m]` 后缀。公开版本把本地绝对路径改为 `TRAJECTORY_CC_BIN` / `TRAJECTORY_CLI_WORKDIR`，省略账户说明；审核逻辑不变。

这是阅读快照，依赖未分发的项目 transport、读者适配器和配置，不是新任务启动说明。入口槽位名称不作为模型身份依据；新增 cc 路由也不算新增模型家族。早期 smoke 的两次超时与稳定生产 273/273 有效回执分开报告。

验证已有数据用 `python scripts/spotcheck_review.py --verify`，不运行本目录生成或审核入口。
