# 视频字幕处理工具

一个桌面Python应用程序，用于从视频中提取语音，生成字幕，并将字幕嵌入到视频中。

## 功能

- 从视频文件中提取音频
- 使用OpenAI Whisper进行语音识别，支持近100种语言
- 自动检测语言，并根据需要将字幕翻译为中文
- 生成SRT格式字幕文件
- 将字幕嵌入到视频中
- 保存带字幕的新视频文件

## 安装和设置

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/video-subtitle-tool.git
cd video-subtitle-tool
```

### 2. 创建虚拟环境（可选但推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装FFmpeg

该工具依赖FFmpeg进行视频和音频处理。

- **Windows**: 
  - 下载FFmpeg: https://ffmpeg.org/download.html
  - 将FFmpeg添加到系统PATH环境变量

- **macOS** (使用Homebrew):
  ```bash
  brew install ffmpeg
  ```

- **Linux**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

### 5. 下载语音模型

运行以下命令下载必要的语音模型:

```bash
python download_models.py
```

## 使用方法

### 基本用法

```bash
python main.py [视频文件路径]
```

如果不提供视频文件路径，默认将处理 `G:\avi1.mp4`。

### 高级选项

```bash
python main.py [视频文件路径] --output [输出文件路径] --lang [目标语言]
```

参数:
- `--output` 或 `-o`: 输出文件路径（默认：在输入文件同目录下创建）
- `--lang` 或 `-l`: 目标语言（默认：'zh-CN'）
- `--hard-sub`: 使用硬字幕嵌入到视频中（默认使用软字幕）
- `--use-ollama`: 使用本地部署的Ollama模型进行翻译
- `--model`: 指定Ollama翻译模型名称（默认：'deepseek-r1'）
- `--output-format`: 指定输出视频格式，可选'mp4'或'mkv'（默认：'mp4'）
- `--subtitle-style`: 指定字幕样式，可选'default'、'outline'、'shadow'或'box'（默认：'default'）

## 处理流程

1. 提取音频: 从视频文件中提取WAV格式的音频
2. 语音识别: 使用Whisper分析音频并生成文本（支持多语言）
3. 创建字幕: 生成带时间戳的SRT格式字幕文件
4. 翻译: 如果检测到非中文字幕，将其翻译成中文
5. 嵌入字幕: 将字幕嵌入到视频中（软字幕或硬字幕）
6. 保存视频: 保存带字幕的新视频文件

## 输出文件

程序会生成以下文件:
1. WAV格式的音频文件
2. 原始转录文本文件
3. 原始语言的SRT字幕文件
4. 翻译后的中文SRT字幕文件
5. 带嵌入字幕的MP4视频文件

## 许可证

MIT 