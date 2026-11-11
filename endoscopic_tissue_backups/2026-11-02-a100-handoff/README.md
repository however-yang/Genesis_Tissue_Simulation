# A100 / Linux 交接源码包

入口：[完整交接与运行命令](../../endoscopic_tissue/docs/handoff-a100-2026-11-02.md)。

此包服务于用户授权的换机开发，所有原生候选仍处于实验状态。未替换默认 DLL，未声称 Linux/A100 已编译或验收通过。

- `source-manifest.json`：四个变体的完整 SHA 清单、父版本和增量文件路径。
- `frozen/`：contact-groups31 重建源码，加 snapshot、boundary、feasible 的字节精确覆盖。来源与对应 Windows 构建记录逐文件匹配；不依赖本机忽略目录。
- `prepare_source.py`：校验后只在 `endoscopic_tissue/outputs/a100-migration/<variant>/source` 生成副本，适配 C 导出、CMake 和 A100 架构；拒绝覆盖既有源目录。
- `check_snapshot.py`：原 S15 smoke/resource 规则的跨平台启动入口，不调用 Windows CPU 拓扑 API。
- `requirements-core.txt`：不含 Genesis 渲染的基础 Python 依赖，不是 Linux 验证锁文件。
- `windows-environment-reference.json`：旧机版本与场景阶段参考。
- `protected-at-handoff.json`、`evidence-manifest.json`：默认保护及本次上传的轻量证据。视频/大回放/快照没有打包。
- `clean-runtime-validation.json`、`transfer-validation.json`：从 Git 暂存内容独立导出后的 94 项 CPU 检查、CLI 入口和四个变体源码重建验证；均不代表 Linux/GPU 已验收。

```bash
python endoscopic_tissue_backups/2026-11-02-a100-handoff/prepare_source.py \
  --variant training-snapshot1 --verify-only
```

完整冻结源码以 `.gitattributes` 的 `-text` 保留原始换行；不要先格式化这些文件再执行 SHA 校验。后续算法修改应放进新实验副本，并建立新的来源清单。
