# HTTP 缓存

HTTP 缓存通过复用已有响应减少网络传输和服务器计算。

- `Cache-Control: max-age=300` 表示响应在 300 秒内可视为新鲜。
- `no-store` 要求缓存不得存储请求或响应内容；`no-cache` 允许存储，但复用前必须验证。
- 服务器可返回 `ETag`，客户端随后用 `If-None-Match` 发起条件请求；内容未变化时服务器返回 `304 Not Modified`。
- `Vary: Accept-Encoding` 表示缓存键必须考虑 `Accept-Encoding` 请求头。
- 共享缓存可使用 `s-maxage`，其优先级高于共享缓存中的 `max-age`。
