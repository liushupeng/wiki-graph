# Node.js 流

流适合增量处理大数据，避免一次性把全部内容放入内存。

- Readable 流的 `highWaterMark` 是内部缓冲阈值，不是严格的内存上限。
- Writable 的 `write()` 返回 `false` 表示生产者应暂停，并等待 `drain` 事件。
- `stream.pipeline()` 负责连接多个流并传播错误与关闭信号。
- object mode 中每个块按对象计数，而不是按字节计数。
- Transform 流同时可读可写，可用于压缩、解析或格式转换。
