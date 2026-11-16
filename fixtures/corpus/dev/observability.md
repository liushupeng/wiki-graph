# 可观测性

可观测性通过运行时信号帮助团队理解系统内部状态。

- 指标（metrics）适合聚合趋势，日志（logs）记录离散事件，追踪（traces）展示请求跨组件路径。
- RED 方法关注请求速率 Rate、错误率 Errors 和持续时间 Duration。
- USE 方法关注资源利用率 Utilization、饱和度 Saturation 和错误 Errors。
- 分布式追踪使用 trace ID 关联同一请求的多个 span。
- 告警应面向可操作症状，并附带负责人、影响和排障入口。
