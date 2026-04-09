# 安装指南

本文档说明如何在本地安装并运行 GPIPA。

## 1. 适用范围

本文档以 Windows 环境为主，适用于以下场景：

- 本地开发
- 本地调试
- 运行 Gradio UI
- 运行完整推理流程

## 2. 环境要求

- Python：`3.11`
- 包管理器：`uv`
- 操作系统：Windows 10 / 11

运行完整推理流程时，还需要以下运行时资源：

- Qwen ASR 模型
- 声学对齐 checkpoint
- `vocab.json`
- FFmpeg shared DLL

## 3. 仓库结构

安装相关文件位于仓库根目录：

- `pyproject.toml`
- `.python-version`
- `uv.lock`
- `config.yml`

运行时目录：

- `vendor/ffmpeg/bin`
- `output/`

文档目录：

- `docs/installation.md`
- `docs/technical.md`

## 4. 安装 uv

推荐使用 `winget`：

```powershell
winget install --id=astral-sh.uv -e
```

验证安装：

```powershell
uv --version
```

## 5. 创建虚拟环境

在仓库根目录执行：

```powershell
uv python install 3.11
uv venv --python 3.11
```

## 6. 安装 Python 依赖

本项目已使用 `uv` 管理依赖，并在 `pyproject.toml` 中定义了依赖分组。

推荐直接安装完整依赖：

```powershell
uv sync --extra full
```

如果仅需要界面相关依赖：

```powershell
uv sync --extra ui
```

## 7. 配置 FFmpeg shared DLL

### 7.1 安装 FFmpeg shared 版本

推荐安装 Gyan 的 shared 构建：

```powershell
winget install -e --id Gyan.FFmpeg.Shared
```

验证命令行可用：

```powershell
ffmpeg -version
where ffmpeg
```

### 7.2 配置项目内 DLL 路径

GPIPA 在 Windows 下会优先尝试从以下目录加载 FFmpeg DLL：

```text
vendor/ffmpeg/bin
```

请将 FFmpeg shared 构建的 `bin` 目录中的 DLL 复制到该目录，至少应包含：

- `avcodec-*.dll`
- `avformat-*.dll`
- `avutil-*.dll`

项目启动时会自动将 `vendor/ffmpeg/bin` 注册到 DLL 搜索路径。

如果希望使用自定义目录，可设置：

```powershell
$env:GPIPA_FFMPEG_BIN="C:\path\to\ffmpeg\bin"
```

## 8. 配置 ASR 模型

默认模型为：

- `Qwen/Qwen3-ASR-0.6B`

如需使用本地模型目录，可设置：

```powershell
$env:GPIPA_QWEN_ASR_MODEL="C:\path\to\Qwen3-ASR-0.6B"
```

可用环境变量如下：

- `GPIPA_QWEN_ASR_MODEL`
- `GPIPA_QWEN_ASR_DEVICE`
- `GPIPA_QWEN_ASR_DTYPE`

常用示例：

```powershell
$env:GPIPA_QWEN_ASR_DEVICE="cpu"
```

## 9. 配置运行时资源

### 9.1 `config.yml`

运行前请检查仓库根目录下的 `config.yml`。

重点配置项：

- `checkpoint_dir`
- `output_dir`
- `root_path`

### 9.2 checkpoint 目录
> [!note]提示
> 如果使用分发的压缩包，那么就已经包含了这些文件，不需要修改任何内容了。

`checkpoint_dir` 目录下应至少包含：

- `best_model.pt`
- `vocab.json`

### 9.3 训练数据目录

>[!NOTE]提示
> 正常使用时，不需要关注本节内容。

`root_path` 仅用于训练或测试导出流程。在推理时，大概率不需要用到这一项。

当前版本中，推理使用的 `use` 路径已不再依赖训练数据目录。也就是说：

- 运行普通推理时，不要求 `root_path` 下存在训练语料
- 运行训练或测试导出时，仍要求 `root_path` 指向完整训练数据集。

训练数据目录是和 DataBaker 一致的。

- `Wave/`
- `ProsodyLabeling/`
- `PhoneLabeling/`

## 10. 启动方式

启动 Gradio UI：

```powershell
uv run python -m gpipa.ui
```
