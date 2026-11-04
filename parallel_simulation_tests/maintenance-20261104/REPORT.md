# 维护验证记录

目录设计和清理说明见 [CODE_LAYOUT](../../docs/CODE_LAYOUT.md)。

- `plan.json`：52 项代码归档清单及原因。
- `archive-summary.json`：归档 SHA256 与尺寸。
- `git-bundle.json` / `bundle-fsck.log`：完整 Git 备份验证。
- `runtime-before.json` / `runtime-after.json`：完整运行文件 SHA256。
- `entry-before.json` / `final-validation.json`：命令与路径核验。
- `cpu-regression.log` / `cpu-regression.json`：136 项现有 CPU 回归全部通过。
- `restore-regression.json`：真实恢复及拒绝覆盖验证。
- `disk-summary.json`：磁盘收益。
- `readme-before/`：原阶段说明备份。
- `cleanup.py`：本次明确清单的操作脚本，已有结果不可重复覆盖。

本轮不运行 GPU 基准或新物理验收。
