# Project continuity

开始研究前阅读 `RESEARCH_HANDOFF.md`；换电脑时先阅读 `README_MIGRATION.md`。
当前重点是 Oracle 验证 ADE 从约 0.1192 m 继续接近 0.05 m。
先核实文件和最新评估，再执行后续工作。已有输出保留，新实验使用新目录。
用户通常自行运行完整数据生成和训练；仅在明确要求执行时启动这些长任务。

## CodeGraph

若仓库根目录存在 `.codegraph/`，定位和理解代码时先用 CodeGraph，再用 rg/文件读取。
若没有该目录，跳过 CodeGraph，不自动建立索引。
