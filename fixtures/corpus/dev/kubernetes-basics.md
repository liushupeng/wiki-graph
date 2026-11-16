# Kubernetes 基础

Kubernetes 通过声明式对象协调应用的期望状态与实际状态。

- Deployment 管理 ReplicaSet，并支持滚动更新无状态工作负载。
- Service 的 ClusterIP 类型提供集群内部稳定虚拟地址。
- readiness probe 失败会让 Pod 从 Service 端点中移除，但不会直接重启容器。
- liveness probe 持续失败可触发 kubelet 重启容器。
- ConfigMap 适合非机密配置；Secret 用于敏感值，但仍需配合静态加密和访问控制。
