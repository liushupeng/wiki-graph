# SQL 事务

事务把多个数据库操作组织成一个逻辑工作单元。

- ACID 分别表示原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation）和持久性（Durability）。
- `READ COMMITTED` 阻止脏读，但同一事务内重复查询仍可能看到不同的已提交结果。
- `REPEATABLE READ` 保证已读取行在事务期间保持可重复读取；不同数据库对幻读的具体处理可能不同。
- `SAVEPOINT checkout_started` 创建事务内保存点，`ROLLBACK TO SAVEPOINT checkout_started` 可回退到该位置。
- 应用遇到死锁时，数据库通常中止其中一个事务；客户端应在有限次数和退避策略下重试整个事务。
