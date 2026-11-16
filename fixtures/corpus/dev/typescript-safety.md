# TypeScript 类型安全

TypeScript 的静态类型用于在运行前发现接口使用错误。

- `unknown` 要求使用者先缩小类型再访问值，比 `any` 更安全。
- 判别联合通常共享一个字面量字段，例如 `kind: "success" | "error"`。
- `never` 可用于穷尽性检查，确保 switch 处理联合类型的每个成员。
- 启用 `strictNullChecks` 后，`null` 和 `undefined` 不再自动兼容其他类型。
- `satisfies` 检查表达式符合目标类型，同时保留表达式更精确的推断类型。
