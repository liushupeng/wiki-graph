# PostgreSQL 索引

B-tree 是 PostgreSQL 的默认索引类型，适合等值与范围查询。GIN 常用于数组、全文检索等包含关系。部分索引只索引满足 WHERE 谓词的行，可减少特定工作负载的索引体积。
