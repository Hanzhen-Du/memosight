"""MemoSight MVP pipeline (阶段一).

守门员触发(mock) → 高清帧(测试图) → 本地 OCR → 打包 → 云端 enricher(mock 生成 tags)
→ SQLite 存储 → 命令行查询。

三个可替换接口，各自独立模块 + 抽象基类：OCR / Enricher / Transport。
"""

__all__ = ["models", "db", "config"]
