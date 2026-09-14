# PyPI 发布准备 - 完成状态

## ✅ 已完成的工作

### 1. **License 更改** ✅
- ✅ 从 PolyForm Noncommercial 1.0.0 改为 MIT License
- ✅ 允许商业使用
- ✅ 与主流开源生态兼容

### 2. **CHANGELOG.md** ✅
- ✅ 创建了详细的 v0.1.0 发布说明
- ✅ 列出所有功能、性能数据、已知限制
- ✅ 遵循 Keep a Changelog 格式

### 3. **README.md 增强** ✅
- ✅ 添加 PyPI badge 和 License badge
- ✅ 在最前面添加 Installation 章节
- ✅ 添加 Quick Start 代码示例（3 种场景）
- ✅ 添加 "When to use PocketLLM" 对比章节
- ✅ 明确说明与 vLLM/SGLang 的差异

### 4. **pyproject.toml 完善** ✅
- ✅ License 改为 MIT
- ✅ 添加 keywords（10+ 关键词）
- ✅ 添加 classifiers（开发状态、Python 版本、主题）
- ✅ 添加作者信息
- ✅ 添加完整的项目 URLs（文档、仓库、Issues、Changelog）
- ✅ 依赖拆分为 optional（triton、cpp）
- ✅ Torch 依赖版本限制为 `<2.7`

### 5. **setup.py 修复** ✅
- ✅ Torch 延迟导入（避免 sdist 构建时需要 torch）
- ✅ CUDA 扩展变为可选（没有 torch 时跳过）
- ✅ 扩展模块条件化定义

### 6. **测试** ✅
- ✅ 创建 `tests/test_install_smoke.py` 基础安装测试
- ✅ 测试导入、API 可用性、版本格式

### 7. **构建验证** ✅
- ✅ 成功构建 sdist: `pocketllm-0.1.0.tar.gz` (1.4 MB)
- ✅ `twine check` 通过验证

### 8. **发布文档** ✅
- ✅ 创建 `docs/PYPI_RELEASE.md` 完整发布指南
- ✅ 包含 Test PyPI 测试流程
- ✅ 包含故障排除章节

---

## 📋 **发布前最终检查清单**

### A. 代码和文档
- [x] License 已更改为 MIT
- [x] CHANGELOG.md 已创建且完整
- [x] README.md 有 Installation 和 Quick Start
- [x] pyproject.toml 元数据完整
- [x] 版本号统一（`__init__.py` 和 `pyproject.toml` 都是 `0.1.0`）

### B. 构建测试
- [x] `python -m build --sdist --no-isolation` 成功
- [x] `twine check dist/*` 通过
- [ ] **本地虚拟环境安装测试**（建议做）
- [ ] **smoke test 运行**（建议做）

### C. Git 状态
- [ ] 所有改动已提交到 git
- [ ] 创建 git tag `v0.1.0`
- [ ] 推送到 GitHub

### D. PyPI 账户
- [ ] 注册 PyPI 账户
- [ ] 生成 API token
- [ ] 配置 `~/.pypirc`

### E. 发布流程
- [ ] （可选但强烈推荐）上传到 Test PyPI 测试
- [ ] 上传到生产 PyPI
- [ ] 推送 git tags
- [ ] 创建 GitHub Release

---

## 🚀 **立即可执行的发布命令**

### 如果要立即发布到 Test PyPI（推荐先测试）：

```bash
# 1. 提交所有改动
git add LICENSE CHANGELOG.md README.md pyproject.toml setup.py \
    tests/test_install_smoke.py docs/PYPI_RELEASE.md
git commit -m "Prepare v0.1.0 release: MIT license, enhanced README, PyPI metadata

- Change license from PolyForm Noncommercial to MIT
- Add Installation and Quick Start to README
- Add CHANGELOG.md with v0.1.0 release notes
- Enhance pyproject.toml with keywords, classifiers, and optional dependencies
- Fix setup.py to make CUDA extensions optional
- Add smoke test for installation verification
- Add PyPI release guide documentation

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"

# 2. 创建 tag
git tag v0.1.0

# 3. 推送（可选，也可以发布成功后再推）
git push origin master
git push origin v0.1.0

# 4. 上传到 Test PyPI
twine upload --repository testpypi dist/*

# 5. 测试安装
python -m venv /tmp/test-pocketllm
source /tmp/test-pocketllm/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pocketllm
python -c "import pocketllm; print(pocketllm.__version__)"
deactivate
```

### 如果测试通过，发布到生产 PyPI：

```bash
# 上传到生产 PyPI
twine upload dist/*

# 创建 GitHub Release
gh release create v0.1.0 \
    --title "PocketLLM v0.1.0 - First PyPI Release" \
    --notes "First public release of PocketLLM on PyPI.

## Highlights
- MIT License (commercial use allowed)
- 4 validated model runtimes (Qwen, DeepSeek-V4, MiniMax, GLM-5.2)
- TP4 inference on consumer GPUs (RTX 2080 Ti tested)
- GGUF Q2/IQ2/FP4/FP8 quantization support
- OpenAI-compatible API server

See CHANGELOG.md for full release notes.

Install:
\`\`\`bash
pip install pocketllm
\`\`\`

🤖 Generated with [Claude Code](https://claude.com/claude-code)" \
    dist/pocketllm-0.1.0.tar.gz
```

---

## ⚠️ **重要提醒**

1. **Test PyPI 测试是强烈推荐的**
   - Test PyPI 是独立的测试环境
   - 可以安全测试整个发布流程
   - 发现问题可以重新上传（生产 PyPI 不允许）

2. **生产 PyPI 发布是不可逆的**
   - 一旦上传，该版本号永久占用
   - 不能删除或覆盖
   - 发现问题只能发布新版本

3. **首次发布建议**
   - 先发布到 Test PyPI
   - 在一个干净的环境测试安装
   - 运行 smoke test
   - 确认无误后再发布到生产环境

---

## 📊 **预期结果**

发布成功后：
- PyPI 页面：https://pypi.org/project/pocketllm/
- 用户可以 `pip install pocketllm`
- GitHub README badge 会显示版本号
- 项目进入 PyPI 搜索索引

---

## 🎯 **下一步（发布后）**

1. **版本号 bump 到开发版**
   ```bash
   # 在 pocketllm/__init__.py 和 pyproject.toml
   version = "0.2.0.dev0"
   ```

2. **监控用户反馈**
   - GitHub Issues
   - PyPI 下载统计

3. **准备 0.1.1 bug fix 版本**（如有需要）

4. **考虑预编译 wheels**（改善用户体验）
   - 使用 `cibuildwheel`
   - 支持 cu118, cu121, cu124
   - 支持 Python 3.10, 3.11, 3.12
