from __future__ import annotations

import json

from langchain_core.tools import tool

_MOCK_TICKETS = {
    "INC-1001": {
        "title": "核心 API 延迟升高",
        "status": "investigating",
        "summary": "过去 15 分钟 p95 延迟超过 2s，已通知值班 SRE。",
        "runbook": "先查依赖服务健康与最近发布。",
    },
    "INC-1002": {
        "title": "磁盘使用率预警",
        "status": "mitigated",
        "summary": "日志分区 85%，已扩容临时卷并轮转日志。",
        "runbook": "确认 logrotate 与监控告警阈值。",
    },
}


@tool
def query_incident_ticket(ticket_id: str) -> str:
    """查询运维工单 Mock 数据（演示 OpenAPI/工单类工具，非 RAG）。"""
    key = ticket_id.strip().upper()
    ticket = _MOCK_TICKETS.get(key)
    if not ticket:
        return json.dumps(
            {"ticketId": key, "found": False, "hint": "示例工单号：INC-1001、INC-1002"},
            ensure_ascii=False,
        )
    return json.dumps({"ticketId": key, "found": True, **ticket}, ensure_ascii=False)