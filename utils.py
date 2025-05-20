import subprocess
import os
import wave
import json
import datetime
import time
import sys
import re
import tempfile
import shutil
from typing import Optional, Tuple, List
import pysrt
from tqdm import tqdm
import whisper
import torch

# 创建临时文件管理类
class TempFileManager:
    def __init__(self):
        self.temp_dir = os.path.join(tempfile.gettempdir(), 'videotoch_temp')
        self.temp_files = []
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def create_temp_file(self, suffix=None):
        """创建临时文件并记录"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=self.temp_dir)
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def cleanup(self):
        """清理所有临时文件"""
        for file in self.temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"警告: 清理临时文件失败 {file}: {e}")
        self.temp_files = []
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"警告: 清理临时目录失败: {e}")

# 创建全局临时文件管理器实例
temp_manager = TempFileManager()

def extract_audio(video_path: str) -> str:
    """提取视频中的音频
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        生成的音频文件路径
    """
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    try:
        print(f"开始从视频提取音频...")
        print(f"输入: {video_path}")
        print(f"输出: {audio_path}")
        
        # 使用ffprobe获取视频时长
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        try:
            duration = float(subprocess.check_output(probe_cmd, timeout=10).decode('utf-8').strip())
            print(f"视频时长: {duration:.2f} 秒")
        except:
            duration = 0
            print("无法获取视频时长，继续处理")
        
        # 构建命令 - 使用更高效的参数配置
        cmd = [
            'ffmpeg', 
            '-i', video_path, 
            '-vn',                # 不处理视频
            '-acodec', 'pcm_s16le', 
            '-ar', '16000',       # 采样率
            '-ac', '1',           # 单声道
            '-y',                 # 覆盖现有文件
            '-v', 'warning',      # 只显示警告和错误
            audio_path
        ]
        
        print("执行命令:", ' '.join(cmd))
        
        # 直接运行命令并捕获输出，设置超时时间
        try:
            # 使用不阻塞的方式运行命令
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                universal_newlines=True, 
                bufsize=1
            )
            
            # 使用带超时的进度条
            max_wait_time = max(180, int(duration * 1.5)) if duration > 0 else 300  # 至少3分钟或视频时长的1.5倍
            print(f"设置最大等待时间: {max_wait_time}秒")
            
            with tqdm(total=100, desc="音频提取进度", unit="%") as pbar:
                start_time = time.time()
                current_progress = 0
                
                # 每0.5秒更新一次进度条
                while process.poll() is None:
                    # 计算已经过的时间占总时间的百分比
                    elapsed = time.time() - start_time
                    if max_wait_time > 0:
                        new_progress = min(int(elapsed / max_wait_time * 100), 95)
                    else:
                        # 如果不知道总时长，使用更保守的进度估计
                        new_progress = min(current_progress + 1, 95)
                    
                    # 只在进度有变化时更新
                    if new_progress > current_progress:
                        pbar.update(new_progress - current_progress)
                        current_progress = new_progress
                    
                    # 检查是否超时
                    if elapsed > max_wait_time:
                        print(f"警告: 处理时间过长，尝试终止进程")
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        break
                    
                    time.sleep(0.5)
                
                # 读取任何剩余输出
                stdout, stderr = process.communicate(timeout=10)
                
                # 检查命令是否成功
                if process.returncode != 0:
                    print(f"错误: FFmpeg退出代码 {process.returncode}")
                    if stderr:
                        print(f"错误信息: {stderr[:500]}..." if len(stderr) > 500 else stderr)
                    return ""
                
                # 完成进度条
                pbar.update(100 - current_progress)
            
        except subprocess.TimeoutExpired:
            print("错误: FFmpeg处理超时")
            process.kill()
            return ""
        except Exception as e:
            print(f"处理过程中出错: {e}")
            if 'process' in locals() and process.poll() is None:
                process.kill()
            return ""
            
        # 验证生成的音频文件
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"✓ 音频提取成功: {audio_path} (大小: {audio_size_mb:.2f} MB)")
            
            # 验证音频文件是否有效
            try:
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    audio_duration = frames / rate
                    print(f"音频时长: {audio_duration:.2f}秒")
                    
                    if audio_duration < 0.5 and duration > 1:
                        print("警告: 生成的音频文件太短，可能未正确提取")
            except Exception as e:
                print(f"警告: 无法验证音频文件: {e}")
                
            return audio_path
        else:
            print("错误: 音频文件不存在或大小为0")
            return ""
    except subprocess.CalledProcessError as e:
        print(f"音频提取失败: {e}")
        return ""

def detect_language(text: str) -> str:
    """检测文本语言
    
    Args:
        text: 文本内容
        
    Returns:
        语言代码 'en'或'zh-CN'
    """
    print("正在检测文本语言...")
    
    # 简单判断是否为中文（包含中文字符）
    chinese_chars = 0
    total_chars = len(text.strip())
    
    if total_chars == 0:
        print("警告: 空文本，默认为英文")
        return 'en'
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_chars += 1
    
    chinese_ratio = chinese_chars / total_chars
    print(f"中文字符比例: {chinese_ratio:.2%}")
    
    if chinese_ratio > 0.1:  # 如果超过10%是中文字符，认为是中文
        print("✓ 检测结果: 中文(zh-CN)")
        return 'zh-CN'
    else:
        print("✓ 检测结果: 英文(en)")
        return 'en'

def enhanced_detect_language(text: str) -> str:
    """增强的语言检测功能，支持更多语言
    
    Args:
        text: 文本内容
        
    Returns:
        语言代码（'en', 'zh-CN', 'ru', 'ja', 等）
    """
    print("正在使用增强版语言检测...")
    
    # 检查文本是否为空
    if not text.strip():
        print("警告: 空文本，默认为英文")
        return 'en'
    
    # 先检查是否含有中文字符
    chinese_chars = 0
    total_chars = len(text.strip())
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_chars += 1
    
    chinese_ratio = chinese_chars / total_chars
    print(f"中文字符比例: {chinese_ratio:.2%}")
    
    if chinese_ratio > 0.1:  # 如果超过10%是中文字符，认为是中文
        print("✓ 检测结果: 中文(zh-CN)")
        return 'zh-CN'
    
    # 对于非中文文本，尝试使用langdetect进行更精确的检测
    try:
        # 动态导入langdetect，避免不必要的依赖
        try:
            from langdetect import detect, LangDetectException
        except ImportError:
            print("警告: langdetect库未安装，使用基本检测")
            print("提示: 可以通过'pip install langdetect'安装")
            # 回退到基本检测
            print("✓ 检测结果(基本): 英文(en)")
            return 'en'
        
        lang_code = detect(text)
        
        # 语言代码映射
        lang_mapping = {
            'en': 'en',      # 英语
            'zh': 'zh-CN',   # 中文
            'ru': 'ru',      # 俄语
            'ja': 'ja',      # 日语
            'ko': 'ko',      # 韩语
            'fr': 'fr',      # 法语
            'de': 'de',      # 德语
            'es': 'es',      # 西班牙语
            'it': 'it',      # 意大利语
            'pt': 'pt',      # 葡萄牙语
            'ar': 'ar',      # 阿拉伯语
            'tr': 'tr',      # 土耳其语
            'vi': 'vi',      # 越南语
            'th': 'th',      # 泰语
        }
        
        detected_lang = lang_mapping.get(lang_code, 'en')  # 默认英语
        print(f"✓ 检测结果(高级): {detected_lang} (原始代码: {lang_code})")
        
        # 显示可能的语言名称
        language_names = {
            'en': '英语',
            'zh-CN': '中文',
            'ru': '俄语',
            'ja': '日语',
            'ko': '韩语',
            'fr': '法语',
            'de': '德语',
            'es': '西班牙语',
            'it': '意大利语',
            'pt': '葡萄牙语',
            'ar': '阿拉伯语',
            'tr': '土耳其语',
            'vi': '越南语',
            'th': '泰语',
        }
        
        if detected_lang in language_names:
            print(f"检测到的语言: {language_names[detected_lang]}")
        
        return detected_lang
        
    except Exception as e:
        print(f"语言检测出错: {e}")
        print("回退到基本检测")
        # 回退到简单检测
        print("✓ 检测结果(基本): 英文(en)")
        return 'en'

def transcribe_audio(audio_path: str) -> Tuple[List[dict], str]:
    """语音识别转文字，使用Whisper自动检测语言并进行转录"""
    
    print("\n开始语音识别...")
    
    # 检查音频文件
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        return [], ""
    
    # 检测GPU是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_info = torch.cuda.get_device_name(0)
        print(f"✓ 检测到GPU: {gpu_info}，将使用GPU加速识别")
    else:
        print("未检测到可用GPU，将使用CPU进行识别")
    
    # 加载Whisper模型
    print(f"加载Whisper语音识别模型...")
    model_name = "medium"  # 改用medium模型提高准确率，可选: tiny, base, small, medium, large
    
    with tqdm(total=100, desc="模型加载进度", unit="%") as pbar:
        # 加载模型可能需要一些时间
        model_load_start = time.time()
        
        try:
            # 分步显示加载进度
            for i in range(5):
                time.sleep(0.1)
                pbar.update(18)  # 分5步加载到90%
            
            model = whisper.load_model(model_name, device=device)
            
            # 加载完成
            pbar.update(10)  # 到达100%
            model_load_time = time.time() - model_load_start
            print(f"✓ Whisper模型'{model_name}'加载完成 (用时: {model_load_time:.2f}秒) 设备: {device}")
        except Exception as e:
            print(f"错误: 模型加载失败: {e}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            return [], ""
    
    print(f"开始处理音频文件: {audio_path}")
    
    # 开始识别
    try:
        # 显示进度条
        with tqdm(total=100, desc="语音识别进度", unit="%") as pbar:
            # 更新进度条 - 这里只是为了UI体验，实际进度由Whisper内部决定
            process_start = time.time()
            
            # 开始处理，设置verbose=True以便在控制台显示Whisper的进度
            print(f"正在识别音频，这可能需要一些时间...")
            
            # 开始转录
            # 获取转录结果
            result = model.transcribe(
                audio_path,
                verbose=False,  # 不显示Whisper的进度输出，我们自己控制显示
                word_timestamps=True,  # 开启词级时间戳
                initial_prompt="这是一段需要准确转录的音频。请仔细听每一个词，确保转录的准确性。"  # 添加提示以提高准确率
            )
            
            # 处理完成
            pbar.update(100)  # 完成进度条
            process_time = time.time() - process_start
            
            detected_language = result.get("language", "未知")
            print(f"✓ 识别完成，用时 {process_time:.2f}秒")
            print(f"✓ 检测到的语言: {detected_language}")
            
        # 提取所需信息
        full_text = result.get("text", "").strip()
        
        # 把Whisper结果转换为与原Vosk格式兼容的格式
        # Whisper的结果格式和Vosk不同，需要转换以兼容现有代码
        transcription_segments = []
        
        # 从Whisper结果中提取segments和words
        segments = result.get("segments", [])
        
        for segment in segments:
            words = segment.get("words", [])
            if words:
                # 创建一个与Vosk结果格式兼容的字典
                segment_dict = {
                    "result": []
                }
                
                for word in words:
                    word_dict = {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0)
                    }
                    segment_dict["result"].append(word_dict)
                
                transcription_segments.append(segment_dict)
                
        # 保存原始文本到临时文件
        transcript_file = temp_manager.create_temp_file(suffix="_transcript.txt")
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"✓ 转录文本已保存到临时文件: {transcript_file}")
        
        # 显示统计信息
        word_count = len(full_text.split())
        segment_count = len(transcription_segments)
        print(f"转录完成:")
        print(f"- 识别单词数: {word_count}")
        print(f"- 转录片段数: {segment_count}")
        print(f"- 文本长度: {len(full_text)}字符")
        
        # 如果结果为空，提供警告
        if not full_text.strip():
            print("警告: 未能识别出任何文本，请检查音频文件质量")
            
        return transcription_segments, full_text.strip()
        
    except Exception as e:
        print(f"转录失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return [], ""

def create_subtitle_file(segments: List[dict], file_path: str) -> bool:
    """根据转录结果创建SRT字幕文件，优化句子划分
    
    Args:
        segments: 包含时间戳的转录结果
        file_path: 输出SRT文件路径
        
    Returns:
        是否成功创建
    """
    try:
        print(f"\n开始创建字幕文件: {file_path}")
        
        if not segments:
            print("错误: 没有转录片段可用")
            return False
        
        segment_count = len(segments)
        print(f"转录片段数: {segment_count}")
        
        # 提取所有单词及其时间戳
        all_words = []
        for segment in segments:
            if 'result' not in segment:
                continue
                
            for word in segment['result']:
                if 'start' in word and 'end' in word and 'word' in word:
                    all_words.append(word)
        
        if not all_words:
            print("错误: 没有有效的单词及时间戳")
            return False
            
        print(f"提取了 {len(all_words)} 个单词")
        
        # 按照时间排序
        all_words.sort(key=lambda x: x['start'])
        
        # 句子结束标记
        sentence_end_markers = ['.', '!', '?', ';', '。', '！', '？', '；', '…']
        clause_markers = [',', ':', '，', '：', '、']
        sentence_starters = ['The', 'A', 'An', 'In', 'On', 'At', 'He', 'She', 'It', 'They', 'We', 'I', 'You', 'This', 'That']
        
        # 合并单词为句子
        sentences = []
        current_sentence = {
            'words': [],
            'start': all_words[0]['start'],
            'end': 0
        }
        
        word_count = 0
        last_end_time = 0
        
        for i, word in enumerate(all_words):
            word_text = word['word'].strip()
            if not word_text:
                continue
                
            word_count += 1
            current_sentence['words'].append(word)
            current_sentence['end'] = word['end']
            last_end_time = word['end']
            
            # 确定是否结束当前句子
            is_sentence_end = False
            
            # 获取当前句子的文本
            current_text = ' '.join(w['word'].strip() for w in current_sentence['words'])
            
            # 情况1: 当前词以句末标点结尾
            if any(word_text.endswith(marker) for marker in sentence_end_markers):
                is_sentence_end = True
            
            # 情况2: 当前词后面是较长停顿（改为1.2秒，在语音停顿和句子完整性之间找平衡）
            elif i < len(all_words)-1 and all_words[i+1]['start'] - word['end'] > 1.2:
                # 确保当前句子有一定长度才断句
                if len(current_text.split()) >= 3:
                    is_sentence_end = True
            
            # 情况3: 句子已经过长或时间过长
            elif len(current_sentence['words']) >= 20 or (word['end'] - current_sentence['start']) > 12:
                if i < len(all_words)-1:
                    next_word = all_words[i+1]['word'].strip()
                    # 检查是否是合适的断句点
                    if (next_word and (
                        next_word[0].isupper() or  # 下一个词是大写开头
                        any(next_word.startswith(starter) for starter in sentence_starters) or  # 下一个词是句子常见开头
                        any(word_text.endswith(marker) for marker in clause_markers)  # 当前词以子句标点结尾
                    )):
                        is_sentence_end = True
            
            # 情况4: 语义完整性检查
            if not is_sentence_end and i < len(all_words)-1:
                next_word = all_words[i+1]['word'].strip()
                # 如果当前句子看起来完整且下一个词明显是新句子的开始
                if (len(current_text.split()) >= 5 and  # 确保有足够的内容
                    next_word and (
                        next_word[0].isupper() or  # 下一个词是大写开头
                        any(next_word.startswith(starter) for starter in sentence_starters)  # 下一个词是句子常见开头
                    )):
                    is_sentence_end = True
            
            # 如果需要结束当前句子
            if is_sentence_end and len(current_sentence['words']) > 0:
                # 确保句子有最小长度
                if len(current_text.split()) >= 3:
                    sentences.append(current_sentence)
                    
                    # 开始新句子（如果还有后续单词）
                    if i < len(all_words)-1:
                        current_sentence = {
                            'words': [],
                            'start': all_words[i+1]['start'],
                            'end': 0
                        }
        
        # 添加最后一个句子（如果有）
        if len(current_sentence['words']) > 0:
            last_text = ' '.join(w['word'].strip() for w in current_sentence['words'])
            if len(last_text.split()) >= 3:  # 确保最后一个句子也有最小长度
                sentences.append(current_sentence)
        
        print(f"根据单词生成了 {len(sentences)} 个句子")
        
        # 创建SRT文件
        subs = pysrt.SubRipFile()
        
        for i, sentence in enumerate(sentences, 1):
            # 提取文本
            text = ' '.join(word['word'] for word in sentence['words']).strip()
            
            # 清理文本，移除多余的空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 获取时间戳
            start_time = sentence['start']
            end_time = sentence['end']
            
            # 确保每个字幕的最短和最长持续时间
            duration = end_time - start_time
            if duration < 1.0:
                end_time = start_time + 1.0
            elif duration > 7.0:  # 限制最长持续时间
                end_time = start_time + 7.0
            
            try:
                # 将秒数转换为时、分、秒、毫秒
                start_seconds = int(start_time)
                start_ms = int((start_time - start_seconds) * 1000)
                start_hours, remainder = divmod(start_seconds, 3600)
                start_minutes, start_seconds = divmod(remainder, 60)
                
                end_seconds = int(end_time)
                end_ms = int((end_time - end_seconds) * 1000)
                end_hours, remainder = divmod(end_seconds, 3600)
                end_minutes, end_seconds = divmod(remainder, 60)
                
                # 使用直接的时间组件创建SubRipTime
                start = pysrt.SubRipTime(
                    hours=start_hours, 
                    minutes=start_minutes, 
                    seconds=start_seconds, 
                    milliseconds=start_ms
                )
                
                end = pysrt.SubRipTime(
                    hours=end_hours, 
                    minutes=end_minutes, 
                    seconds=end_seconds, 
                    milliseconds=end_ms
                )
                
                # 创建字幕项
                sub = pysrt.SubRipItem(i, start, end, text)
                subs.append(sub)
                
            except Exception as e:
                print(f"警告: 创建字幕项失败: {e}, 跳过此句")
                continue
        
        # 如果没有创建任何字幕，则返回失败
        if len(subs) == 0:
            print("错误: 未能创建任何字幕项")
            return False
            
        # 写入SRT文件前显示统计信息
        print(f"字幕统计:")
        print(f"- 总字幕条数: {len(subs)}")
        print(f"- 总单词数: {word_count}")
        
        # 写入SRT文件
        print(f"正在保存字幕文件...")
        subs.save(file_path, encoding='utf-8')
        
        # 验证文件是否成功保存
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"✓ 字幕文件成功保存: {file_path}")
            print(f"  文件大小: {os.path.getsize(file_path)} 字节")
            return True
        else:
            print(f"错误: 字幕文件保存失败或文件为空")
            return False
        
    except Exception as e:
        print(f"创建字幕文件失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return False

def translate_text(text: str, source_lang: str = None, target_lang: str = 'zh-CN', show_progress: bool = True) -> str:
    """翻译文本"""
    if not text.strip():
        print("警告: 空文本，无需翻译")
        return ""
    
    print(f"翻译文本片段 ({len(text)} 字符)...")
    sample = text[:50] + "..." if len(text) > 50 else text
    print(f"样本: \"{sample}\"")
        
    # 如果未指定源语言，自动检测
    if source_lang is None:
        source_lang = detect_language(text)
    
    # 如果源语言和目标语言相同，无需翻译
    if source_lang == target_lang:
        print(f"检测到文本已经是{target_lang}语言，无需翻译")
        return text
    
    print(f"翻译方向: {source_lang} → {target_lang}")
    
    # 尝试使用多种翻译API，按优先级排序
    translate_methods = [
        "ollama",           # 首先使用本地Ollama模型
    ]
    
    translated_text = ""
    
    for method in translate_methods:
        try:
            if method == "ollama":
                translated_text = translate_text_with_ollama(
                    text,
                    model_name="huihui_ai/qwen3-abliterated",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    show_progress=show_progress
                )
            
            if translated_text:
                # 显示翻译结果预览
                result_sample = translated_text[:50] + "..." if len(translated_text) > 50 else translated_text
                print(f"✓ 翻译完成(使用{method}): \"{result_sample}\"")
                
                # 保存翻译结果到临时文件
                if len(text) > 100:  # 只为较长文本保存文件
                    temp_file = temp_manager.create_temp_file(suffix="_translated.txt")
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write(translated_text)
                    print(f"翻译文本已保存到临时文件: {temp_file}")
                
                return translated_text
        except Exception as e:
            print(f"{method}翻译错误: {e}")
            print(f"尝试下一个翻译方法...")
            continue
    
    # 所有方法都失败了
    print("所有翻译方法都失败了，返回原文")
    return text

def translate_text_with_ollama(text: str, model_name: str = "huihui_ai/qwen3-abliterated", source_lang: str = None, target_lang: str = 'zh-CN', show_progress: bool = True, base_url: str = "http://localhost:11434") -> str:
    """使用本地部署的Ollama模型进行翻译
    
    Args:
        text: 要翻译的文本
        model_name: Ollama模型名称
        source_lang: 源语言（如果为None则自动检测）
        target_lang: 目标语言
        show_progress: 是否显示进度信息
        base_url: Ollama服务器地址
        
    Returns:
        翻译后的文本
    """
    if not text.strip():
        print("警告: 空文本，无需翻译")
        return ""
    
    print(f"\n使用Ollama模型 {model_name} 翻译文本 ({len(text)} 字符)...")
    sample = text[:50] + "..." if len(text) > 50 else text
    print(f"样本: \"{sample}\"")
    
    # 如果未指定源语言，自动检测
    if source_lang is None:
        source_lang = enhanced_detect_language(text)
    
    # 如果源语言和目标语言相同，无需翻译
    if source_lang == target_lang:
        print(f"检测到文本已经是{target_lang}语言，无需翻译")
        return text
    
    print(f"翻译方向: {source_lang} → {target_lang}")
    
    try:
        import requests
        import json
        
        # 构建提示词
        prompt = f"""请将以下{source_lang}翻译成中文。
重要说明：
1. 只输出翻译结果，不要有任何解释、思考过程或额外内容
2. 不要使用引号包裹翻译结果
3. 不要添加"翻译:"或类似前缀
4. 保留原文的标点符号风格

原文:
{text}
/no_think"""

        print("\n发送请求...")
        if show_progress:
            print(f"提示词:\n{prompt}")
        
        # 使用自定义服务器地址
        response = requests.post(
            f'{base_url}/api/generate',
            json={
                'model': model_name,
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )
        
        print(f"\nAPI响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if show_progress:
                print("\n完整响应:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
            translated_text = result.get('response', '').strip()
            
            # 清理翻译结果，去除思考过程和标签
            translated_text = clean_ollama_output(translated_text)
            
            if translated_text:
                result_sample = translated_text[:50] + "..." if len(translated_text) > 50 else translated_text
                print(f"\n翻译结果: {result_sample}")
                return translated_text
            else:
                print("警告: 翻译结果为空")
                return text
        else:
            print(f"错误: API调用失败，状态码 {response.status_code}")
            if response.text:
                print(f"错误信息: {response.text}")
            return text
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n连接错误: {e}")
        print("请检查:")
        print(f"1. Ollama服务是否已启动")
        print(f"2. 是否可以访问 {base_url}")
        print("3. 防火墙设置是否允许访问该端口")
        return text
    except requests.exceptions.Timeout:
        print("\n错误: API请求超时")
        return text
    except Exception as e:
        print(f"\n其他错误: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return text

def clean_ollama_output(text: str) -> str:
    """清理Ollama输出中的思考过程和额外内容
    
    Args:
        text: Ollama模型的原始输出
        
    Returns:
        清理后的文本
    """
    import re
    
    # 如果文本为空，直接返回
    if not text or not text.strip():
        return ""
    
    # 移除可能的引号包裹
    text = text.strip('"\'')
    
    # 移除<think></think>标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # 移除单独的<think>或</think>标签
    text = text.replace('<think>', '').replace('</think>', '').strip()
    
    # 开始多阶段清理处理
    
    # 第1阶段：处理前缀
    # 更全面的前缀列表
    prefixes = [
        "翻译:", "翻译：", "翻译结果:", "翻译结果：", "中文翻译:", "中文翻译：",
        "Translation:", "Translated text:", "Result:", "Chinese translation:",
        "以下是翻译:", "以下是翻译：", "以下是中文翻译:", "以下是中文翻译：",
        "Here is the translation:", "Here's the translation:", "The translation is:"
    ]
    
    # 检查是否存在前缀并移除
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # 第2阶段：处理思考过程
    # 扩展匹配模式
    thought_patterns = [
        # 匹配开头的思考过程
        r"^(我需要将.*?翻译成.*?。)\s*",     # "我需要将...翻译成..."
        r"^(我要把.*?翻译成.*?。)\s*",       # "我要把...翻译成..."
        r"^(我将把.*?翻译成.*?。)\s*",       # "我将把...翻译成..."
        r"^(首先，我.*?)\s*",               # "首先，我..."
        r"^(我会.*?翻译.*?)\s*",            # "我会...翻译..."
        r"^(让我.*?翻译.*?)\s*",            # "让我...翻译..."
        r"^(这是.*?的翻译[：:])\s*",        # "这是...的翻译："
        r"^(接下来我会.*?)\s*",             # "接下来我会..."
        r"^(我将.*?翻译为.*?)\s*",          # "我将...翻译为..."
        r"^(I need to translate.*?)\s*",    # 英文思考过程
        r"^(I will translate.*?)\s*",       # 英文思考过程
        r"^(Let me translate.*?)\s*",       # 英文思考过程
        
        # 匹配整个句子的思考过程
        r"^[^，。,.:;；]*?(我会|让我|接下来)[^，。,.:;；]*?翻译[^，。,.:;；]*?。", # 中文句子级思考过程
        r"^[^,.;:]*?(I will|Let me|I'll)[^,.;:]*?translate[^,.;:]*?\.", # 英文句子级思考过程
    ]
    
    # 移除匹配到的思考过程
    for pattern in thought_patterns:
        text = re.sub(pattern, "", text)
    
    # 第3阶段：移除括号和标记中的内容
    bracket_patterns = [
        r"\(.*?\)",                # 移除所有英文圆括号内容
        r"（.*?）",                 # 移除所有中文圆括号内容
        r"<.*?>",                  # 移除所有尖括号内容
        r"【.*?】",                 # 移除所有中文方括号内容
        r"\[.*?\]",                # 移除所有英文方括号内容
        r"『.*?』",                 # 移除所有中文书名号内容
        r"「.*?」",                 # 移除所有中文引号内容
        r"#.*?#",                  # 移除井号之间的内容
    ]
    
    for pattern in bracket_patterns:
        text = re.sub(pattern, "", text)
    
    # 第4阶段：处理多行文本
    lines = text.split('\n')
    
    # 过滤掉明显是注释或思考的行
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 跳过明显是思考过程的行
        if any(line.startswith(marker) for marker in ["注：", "注意：", "Note:", "PS:", "P.S."]):
            continue
            
        # 跳过非常短的行（通常是分隔符或无意义内容）
        if len(line) < 2:
            continue
            
        filtered_lines.append(line)
    
    # 如果过滤后只剩一行，直接返回
    if len(filtered_lines) == 1:
        return filtered_lines[0]
    
    # 如果有多行，尝试找到最可能的翻译内容
    # 通常翻译内容是最长的连续文本块
    if filtered_lines:
        # 找到最长的行，这通常是翻译内容
        longest_line = max(filtered_lines, key=len)
        if len(longest_line) > 20:  # 确保长度足够算作翻译内容
            return longest_line
        
        # 否则返回所有过滤后的行
        return "\n".join(filtered_lines)
    
    # 第5阶段：处理尾部内容
    # 移除末尾的注释或说明
    end_markers = [
        "希望这个翻译对你有帮助", "这是翻译后的文本", "这是译文", "以上是翻译",
        "请让我知道", "如有需要", "如果有任何问题", "如需进一步帮助",
        "I hope this translation is helpful", "Here's the translation", 
        "Let me know if", "If you need"
    ]
    
    for marker in end_markers:
        if marker in text:
            text = text.split(marker)[0].strip()
    
    # 最后清理和格式化
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 最后返回清理后的文本
    return text

def translate_subtitle_file(srt_path: str, target_lang: str = 'zh-CN', use_ollama: bool = True, model_name: str = "huihui_ai/qwen3-abliterated", retry_translate: bool = False, base_url: str = "http://localhost:11434") -> str:
    """翻译SRT字幕文件，使用单一进度条
    
    Args:
        srt_path: SRT文件路径
        target_lang: 目标语言
        use_ollama: 是否使用Ollama模型
        model_name: Ollama模型名称
        retry_translate: 是否在翻译失败时自动重试
        base_url: Ollama服务器地址
        
    Returns:
        翻译后的SRT文件路径
    """
    try:
        print(f"\n开始翻译字幕文件: {srt_path}")
        
        if not os.path.exists(srt_path):
            print(f"错误: 字幕文件不存在: {srt_path}")
            return ""
            
        # 读取原始字幕
        print("读取字幕文件...")
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
        except UnicodeDecodeError:
            # 尝试其他编码
            print("UTF-8编码读取失败，尝试其他编码...")
            for encoding in ['gbk', 'latin-1', 'cp1252']:
                try:
                    subs = pysrt.open(srt_path, encoding=encoding)
                    print(f"成功使用{encoding}编码读取字幕")
                    break
                except:
                    continue
        
        print(f"字幕条数: {len(subs)}")
        
        if len(subs) == 0:
            print("警告: 字幕文件为空")
            return srt_path
        
        # 初始化翻译结果列表
        translated_texts = [None] * len(subs)
        
        # 对每个字幕条目单独进行语言检测和翻译
        with tqdm(total=len(subs), desc="字幕翻译总进度", unit="条") as pbar:
            for i, sub in enumerate(subs):
                # 清理文本
                text = sub.text.strip()
                if not text:
                    translated_texts[i] = text
                    pbar.update(1)
                    continue
                
                # 对每个字幕条目单独进行语言检测
                source_lang = enhanced_detect_language(text)
                
                # 如果是目标语言，则不需要翻译
                if source_lang == target_lang:
                    translated_texts[i] = text
                    pbar.update(1)
                    continue
                
                # 尝试翻译
                max_retries = 5 if retry_translate else 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        if use_ollama:
                            # 使用Ollama翻译
                            translated_text = translate_text_with_ollama(
                                text,
                                model_name=model_name,
                                source_lang=source_lang,
                                target_lang=target_lang,
                                show_progress=False,
                                base_url=base_url
                            )
                        else:
                            # 使用增强的Google翻译
                            translated_text = translate_text(
                                text,
                                source_lang=source_lang,
                                target_lang=target_lang,
                                show_progress=False
                            )
                        
                        if translated_text:
                            translated_texts[i] = translated_text.strip()
                            success = True
                        else:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"翻译失败，等待2秒后重试...")
                                time.sleep(2)
                    except Exception as e:
                        print(f"翻译字幕 {i+1} 时出错: {e}")
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"等待2秒后重试...")
                            time.sleep(2)
                
                if not success:
                    print(f"警告: 字幕 {i+1} 翻译失败，使用原文")
                    translated_texts[i] = text
                
                pbar.update(1)
        
        # 更新字幕文本
        for i, text in enumerate(translated_texts):
            if text is not None:
                subs[i].text = text
            
        # 保存翻译后的字幕
        translated_path = os.path.splitext(srt_path)[0] + f"_{target_lang}.srt"
        print(f"正在保存翻译后的字幕: {translated_path}")
        
        try:
            subs.save(translated_path, encoding='utf-8')
        except Exception as e:
            print(f"保存字幕时出错: {e}")
            # 尝试直接写入文件
            with open(translated_path, 'w', encoding='utf-8') as f:
                for i, sub in enumerate(subs):
                    f.write(f"{i+1}\n")
                    f.write(f"{sub.start} --> {sub.end}\n")
                    f.write(f"{sub.text}\n\n")
            print("使用备用方法保存字幕")
        
        # 验证保存结果
        if os.path.exists(translated_path) and os.path.getsize(translated_path) > 0:
            print(f"✓ 翻译后的字幕已保存: {translated_path}")
            print(f"  文件大小: {os.path.getsize(translated_path)} 字节")
            return translated_path
        else:
            print("错误: 翻译后的字幕保存失败或文件为空")
            return srt_path
        
    except Exception as e:
        print(f"翻译字幕文件失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return srt_path

def embed_subtitles(video_path: str, subtitle_path: str, output_path: Optional[str] = None, options: dict = None):
    """将SRT字幕嵌入视频
    
    Args:
        video_path: 原始视频路径
        subtitle_path: SRT字幕文件路径
        output_path: 输出文件路径(可选)
        options: 嵌入选项，包括:
            - hard_sub: 是否使用硬字幕
            - format: 输出格式 ('mp4' 或 'mkv')
            - subtitle_style: 字幕样式 ('default', 'outline', 'shadow', 'box')
    """
    # 设置默认选项
    if options is None:
        options = {
            'hard_sub': False,  # 默认使用软字幕
            'format': 'mp4',
            'subtitle_style': 'default'
        }
    
    # 确保选项包含所有必要的键
    options.setdefault('hard_sub', False)
    options.setdefault('format', 'mp4')
    options.setdefault('subtitle_style', 'default')
    
    # 设置输出路径
    if not output_path:
        base_name = os.path.splitext(video_path)[0]
        extension = '.mkv' if options['format'] == 'mkv' else '.mp4'
        output_path = f"{base_name}_subtitled{extension}"
    
    print(f"\n开始将字幕嵌入视频...")
    print(f"视频文件: {video_path}")
    print(f"字幕文件: {subtitle_path}")
    print(f"输出文件: {output_path}")
    print(f"嵌入模式: {'硬字幕' if options['hard_sub'] else '软字幕'}")
    print(f"输出格式: {options['format'].upper()}")
    print(f"字幕样式: {options['subtitle_style']}")
    
    # 检查输入文件
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return False
        
    if not os.path.exists(subtitle_path):
        print(f"错误: 字幕文件不存在: {subtitle_path}")
        return False
    
    # 检查是否存在ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        print("错误: 未找到ffmpeg，请确保ffmpeg已安装并添加到PATH")
        return False
    
    # 获取视频信息
    print("获取视频信息...")
    video_info = {}
    try:
        # 获取视频时长
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration = float(subprocess.check_output(probe_cmd, timeout=10).decode('utf-8').strip())
        video_info['duration'] = duration
        print(f"视频时长: {duration:.2f} 秒")
        
        # 获取视频分辨率
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        resolution = subprocess.check_output(probe_cmd, timeout=10).decode('utf-8').strip()
        video_info['resolution'] = resolution
        print(f"视频分辨率: {resolution}")
        
    except Exception as e:
        print(f"警告: 无法获取完整视频信息: {e}")
    
    # 检查和修复字幕文件编码
    print("检查字幕文件编码...")
    subtitle_fixed_path = subtitle_path
    try:
        # 读取字幕文件
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            subtitle_content = f.read()
            
        # 检查是否包含中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in subtitle_content)
        if has_chinese:
            print("检测到中文字幕")
            
            # 将字幕内容写入临时文件，确保UTF-8编码无BOM
            fixed_path = subtitle_path + ".fixed.srt"
            with open(fixed_path, 'w', encoding='utf-8-sig') as f:
                f.write(subtitle_content)
            
            subtitle_fixed_path = fixed_path
            print(f"已创建编码修复的字幕文件: {subtitle_fixed_path}")
    except UnicodeDecodeError:
        # 如果UTF-8读取失败，尝试其他编码
        try:
            with open(subtitle_path, 'r', encoding='gbk') as f:
                subtitle_content = f.read()
            
            # 转换为UTF-8编码
            fixed_path = subtitle_path + ".fixed.srt"
            with open(fixed_path, 'w', encoding='utf-8-sig') as f:
                f.write(subtitle_content)
            
            subtitle_fixed_path = fixed_path
            print(f"已将字幕文件从GBK转换为UTF-8: {subtitle_fixed_path}")
        except Exception as e:
            print(f"警告: 字幕编码修复失败: {e}")
    except Exception as e:
        print(f"警告: 字幕文件检查失败: {e}")
    
    # 初始化字体文件变量
    font_file = None
    font_path = None
    
    # 如果是硬字幕，需要准备字体
    if options['hard_sub']:
        print("查找系统中的中文字体...")
        try:
            # 尝试找到系统中的中文字体
            font_dirs = [
                "C:\\Windows\\Fonts",  # Windows
                "/usr/share/fonts",    # Linux
                "/System/Library/Fonts" # macOS
            ]
            
            chinese_font_names = [
                "simhei.ttf", "SimHei.ttf",  # 黑体
                "msyh.ttc", "msyhbd.ttc",    # 微软雅黑
                "simkai.ttf",                # 楷体
                "simsun.ttc", "simsunb.ttf", # 宋体
                "STHeiti Light.ttc", "STHeiti Medium.ttc", # macOS中文字体
                "WenQuanYi Micro Hei.ttf", "wqy-microhei.ttc" # Linux中文字体
            ]
            
            for font_dir in font_dirs:
                if not os.path.exists(font_dir):
                    continue
                    
                for font_name in chinese_font_names:
                    test_path = os.path.join(font_dir, font_name)
                    if os.path.exists(test_path):
                        font_path = test_path
                        break
                        
                if font_path:
                    break
                    
            if font_path:
                print(f"找到中文字体: {font_path}")
                
                # 创建临时字体配置文件
                font_file = "fontconfig.txt"
                with open(font_file, 'w', encoding='utf-8') as f:
                    f.write(f"[Script Info]\n")
                    f.write(f"ScriptType: v4.00+\n")
                    f.write(f"PlayResX: 384\n")
                    f.write(f"PlayResY: 288\n")
                    f.write(f"ScaledBorderAndShadow: yes\n\n")
                    f.write(f"[V4+ Styles]\n")
                    f.write(f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                    
                    # 根据样式选择不同的字幕格式
                    if options['subtitle_style'] == 'outline':
                        # 轮廓样式：白色字体，黑色边框，无阴影
                        f.write(f"Style: Default,{os.path.basename(font_path)},24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n")
                    elif options['subtitle_style'] == 'shadow':
                        # 阴影样式：白色字体，无边框，灰色阴影
                        f.write(f"Style: Default,{os.path.basename(font_path)},24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,0,2,2,10,10,10,1\n")
                    elif options['subtitle_style'] == 'box':
                        # 方框样式：白色字体，半透明背景
                        f.write(f"Style: Default,{os.path.basename(font_path)},24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,0,0,2,10,10,10,1\n")
                    else:
                        # 默认样式：白色字体，黑色边框，阴影
                        f.write(f"Style: Default,{os.path.basename(font_path)},24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n")
            else:
                print("警告: 未找到中文字体，将使用默认字体")
        except Exception as e:
            print(f"警告: 创建字体配置失败: {e}")
    
    # 确保路径被正确处理
    video_path = os.path.normpath(video_path)
    subtitle_fixed_path = os.path.normpath(subtitle_fixed_path)
    output_path = os.path.normpath(output_path)
    
    # 设置命令行参数
    if options['hard_sub']:
        # 硬字幕 - 使用字幕滤镜
        style_params = "FontSize=24,PrimaryColour=&H00FFFFFF"  # 基本参数：字体大小和颜色
        
        # 根据样式添加额外参数
        if options['subtitle_style'] == 'outline':
            style_params += ",OutlineColour=&H00000000,Outline=2,Shadow=0"
        elif options['subtitle_style'] == 'shadow':
            style_params += ",OutlineColour=&H00000000,BackColour=&H80000000,Outline=0,Shadow=2"
        elif options['subtitle_style'] == 'box':
            style_params += ",BackColour=&H80000000,BorderStyle=3,Outline=0,Shadow=0"
        else:  # default
            style_params += ",OutlineColour=&H00000000,BackColour=&H00000000,Outline=1,Shadow=1"
            
        # 设置字幕滤镜
        subtitle_filter = None
        if font_file and os.path.exists(font_file) and font_path:
            # 使用自定义字体配置
            # 在Windows上，需要处理路径中的特殊字符，使用单引号包裹整个表达式，内部使用双引号
            subtitle_path_escaped = subtitle_fixed_path.replace('\\', '\\\\').replace(':', '\\:')
            subtitle_filter = f"subtitles='{subtitle_path_escaped}':fontsdir=.:force_style='FontName={os.path.basename(font_path)},{style_params}'"
        else:
            # 使用默认配置
            subtitle_path_escaped = subtitle_fixed_path.replace('\\', '\\\\').replace(':', '\\:')
            subtitle_filter = f"subtitles='{subtitle_path_escaped}':force_style='FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,Outline=1,Shadow=1'"
        
        # 构建硬字幕命令
        cmd = [
            'ffmpeg', 
            '-i', video_path,
            # 强制覆盖输出文件
            '-y',
            # 设置较高的日志级别以减少输出
            '-v', 'warning',
            # 使用字幕滤镜
            '-vf', subtitle_filter,
            # 视频和音频编码
            '-c:v', 'libx264', 
            '-c:a', 'aac', 
            # 视频质量 (23是合理的值，较低的值质量更好但文件更大)
            '-crf', '23',
            # 增加处理速度的预设
            '-preset', 'fast',
            # 输出文件
            output_path
        ]
    else:
        # 使用用户提供的成功命令格式作为参考
        # 成功命令: ffmpeg -i "G:\avi1.mp4" -i "G:\avi1_original_zh-CN.srt" -c copy -c:s mov_text -metadata:s:s:0 language=chi -disposition:s:0 default "G:\avi1_subtitled.mp4"
        if options['format'] == 'mkv':
            # MKV格式软字幕
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', subtitle_fixed_path,
                # 复制视频和音频流
                '-c', 'copy',
                # 设置字幕格式
                '-c:s', 'srt',
                # 设置默认字幕
                '-metadata:s:s:0', 'language=chi',
                '-disposition:s:0', 'default',
                # 强制覆盖
                '-y',
                # 输出文件
                output_path
            ]
        else:
            # MP4格式软字幕 - 使用用户确认可用的命令格式
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', subtitle_fixed_path,
                # 复制视频和音频流
                '-c', 'copy',
                # 设置字幕编码
                '-c:s', 'mov_text',
                # 设置字幕默认显示
                '-metadata:s:s:0', 'language=chi',
                '-disposition:s:0', 'default',
                # 强制覆盖
                '-y',
                # 输出文件
                output_path
            ]
    
    # 执行命令
    print("执行命令:", ' '.join(cmd))
    
    try:
        # 确定执行时间上限
        max_time = max(300, int(video_info.get('duration', 0) * 2))
        print(f"设置最大执行时间: {max_time}秒")

        # 在Windows上处理命令参数，确保路径正确
        if os.name == 'nt':  # Windows系统
            cmd = [f'"{arg}"' if os.path.exists(arg) or (' ' in arg and not arg.startswith('-')) else arg for arg in cmd]
            shell = True
        else:
            shell = False

        # 实时输出ffmpeg信息
        process = subprocess.Popen(
            cmd if not shell else ' '.join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并输出
            universal_newlines=True,
            shell=shell,  # Windows下使用shell
            bufsize=1
        )

        print("\n开始处理视频，请稍候...")
        # 实时读取并打印输出
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # 检查是否包含进度信息
                if "frame=" in line or "video:" in line or "audio:" in line:
                    print(line.strip())
                # 检查是否是最终的编码信息
                elif "muxing overhead" in line:
                    print(line.strip())
                    print("\n✓ 视频处理完成！")

        # 等待进程结束
        return_code = process.wait(timeout=max_time)

        if return_code == 0:
            print("✓ 字幕嵌入成功")
            success = True
        else:
            print(f"✗ 字幕嵌入失败，返回码: {return_code}")
            success = False

    except subprocess.TimeoutExpired:
        print(f"错误: 处理超时")
        process.kill()
        success = False
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        if 'process' in locals():
            process.kill()
        success = False
    
    # 检查输出文件
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:  # 文件大于1MB
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"输出文件: {output_path}")
        print(f"文件大小: {size_mb:.2f} MB")
        success = True
    else:
        print(f"警告: 输出文件不存在或太小，嵌入可能失败")
        success = False
    
    # 清理临时文件
    if subtitle_fixed_path != subtitle_path and os.path.exists(subtitle_fixed_path):
        try:
            os.remove(subtitle_fixed_path)
            print(f"已删除临时字幕文件: {subtitle_fixed_path}")
        except:
            pass
            
    if font_file and os.path.exists(font_file):
        try:
            os.remove(font_file)
            print(f"已删除临时字体配置文件: {font_file}")
        except:
            pass
    
    return success