> GitHub 仅发布服务器归档说明、清单和恢复工具，不上传 tar/bundle。要从 Git 历史恢复已发布的旧源码，可运行 `git restore --source d201cfd5543c6caa27556aae6dbfe5fbfb191101 -- 路径`；下文 tar/bundle 恢复命令用于已有归档的服务器副本。

# 历史归档与恢复

当前源码从 [../current/](../current/README.md) 进入。这里保存淘汰代码与发布历史，避免旧候选继续混在日常开发目录。

- `retired-code-20261104.tar.gz`：本轮归档的中间/失败候选和一次性脚本，保留原项目相对路径、内容、权限与链接。
- `retired-code-20261104-manifest.json`：每个文件的 SHA256、原路径和归档原因。压缩后逐成员校验，移除前再次检查原文件未变。
- `publications/genesis-history-through-d201cfd.bundle`：包含 GitHub 归档提交 `d201cfd` 及祖先的完整 Git bundle，已在空 Git 仓库导入并通过 `git fsck --full`。它不依赖已清理的导出副本。
- `publications/github-publish-receipt-20261104.json`：上一轮 GitHub 发布回执原件；其中旧工作副本/增量 bundle 的位置已由本次完整 bundle 替代。
- `deployment-20261102-README.md`：最初部署说明原件，作为历史记录保留。

## 按需恢复历史代码

在项目根目录列出可恢复项，使用清单中的完整路径：

```bash
python3 archive/restore.py --list
python3 archive/restore.py --path parallel_simulation_tests/20261103-s18-s25/profiling/native-core-25hz-20261104/native-tool-v2 --dry-run
```

确认需要恢复时，去掉 `--dry-run`。脚本校验归档和文件内容，将所选项恢复到原路径；现有目标一律拒绝覆盖。它不启动 GPU、构建或仿真。

旧实验构建/生成脚本、旧计划文件有时引用本次归档的候选。重跑那一轮历史实验前，先按原路径恢复对应候选及脚本。当前保留 V8 的运行、构建和回归入口无需恢复淘汰版本。原冻结 `delivery/manifest.json` 的 harness 清单仍记录历史脚本；没有改写这些历史散列。

归档保留失败原因和原始失败源码（包括未保留的一次性生成器语法错误），不是把它们标为可用版本。所有原始实验结果、失败快照、拓扑最小复现和日志仍在原测试目录。

若要恢复完整 Git 工作副本：

```bash
git clone archive/publications/genesis-history-through-d201cfd.bundle /tmp/genesis-history
```

GitHub 上已发布的归档及提交历史没有因本次本地整理而改动。不得把归档候选当作当前交互实现。
