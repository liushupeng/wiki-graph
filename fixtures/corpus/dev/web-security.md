# Web 安全基础

Web 应用应在输入、身份、会话和输出边界实施分层防护。

- 防御 SQL 注入应使用参数化查询，不应拼接不可信输入。
- 输出到 HTML 时必须按上下文编码，以降低跨站脚本（XSS）风险。
- SameSite Cookie 可减少跨站请求携带 Cookie 的场景，是 CSRF 防护的一层。
- Content Security Policy（CSP）可限制脚本、样式和其他资源的允许来源。
- 密码存储应使用带盐、成本可调的专用哈希算法，例如 Argon2id，而不是通用快速哈希。
