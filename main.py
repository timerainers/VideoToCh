import argparse
import os
import sys
import time
from utils import extract_audio, transcribe_audio, translate_subtitle_file, embed_subtitles, create_subtitle_file, temp_manager

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='视频字幕提取、翻译和嵌入工具')
    
    # 输入输出参数
    parser.add_argument('input', help='输入视频文件路径')
    parser.add_argument('-o', '--output', help='输出视频文件路径(可选)')
    parser.add_argument('--srt', help='输出SRT字幕文件路径(可选)')
    parser.add_argument('--audio', help='输出音频文件路径(可选)')
    
    # 功能控制参数
    parser.add_argument('--extract-only', action='store_true', help='仅提取音频')
    parser.add_argument('--transcribe-only', action='store_true', help='仅转录字幕')
    parser.add_argument('--translate-only', action='store_true', help='仅翻译字幕')
    parser.add_argument('--embed-only', action='store_true', help='仅嵌入字幕')
    
    # 字幕参数
    parser.add_argument('--target-lang', default='zh-CN', help='目标语言(默认: zh-CN)')
    parser.add_argument('--use-ollama', action='store_true', help='使用本地Ollama模型进行翻译')
    parser.add_argument('--model-name', default='huihui_ai/qwen3-abliterated', help='Ollama模型名称')
    parser.add_argument('--retry-translate', action='store_true', help='翻译失败时自动重试')
    
    # 字幕嵌入参数
    parser.add_argument('--hard-sub', action='store_true', help='使用硬字幕')
    parser.add_argument('--format', choices=['mp4', 'mkv'], default='mp4', help='输出视频格式(默认: mp4)')
    parser.add_argument('--subtitle-style', choices=['default', 'outline', 'shadow', 'box'], 
                      default='default', help='字幕样式(默认: default)')
    
    args = parser.parse_args()
    
    try:
        # 显示参数信息
        print("\n运行参数:")
        print(f"输入文件: {args.input}")
        if args.output:
            print(f"输出文件: {args.output}")
        if args.srt:
            print(f"字幕文件: {args.srt}")
        if args.audio:
            print(f"音频文件: {args.audio}")
        print(f"目标语言: {args.target_lang}")
        if args.use_ollama:
            print(f"使用Ollama模型: {args.model_name}")
        if args.retry_translate:
            print("启用翻译重试")
        if args.hard_sub:
            print("使用硬字幕")
        print(f"输出格式: {args.format}")
        print(f"字幕样式: {args.subtitle_style}")
        
        # 检查输入文件
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            return 1
            
        # 设置输出路径
        if not args.output:
            base_name = os.path.splitext(args.input)[0]
            args.output = f"{base_name}_subtitled.{args.format}"
            
        if not args.srt:
            args.srt = os.path.splitext(args.input)[0] + "_original.srt"
            
        if not args.audio:
            args.audio = os.path.splitext(args.input)[0] + ".wav"
            
        # 执行流程
        audio_path = ""
        srt_path = ""
        translated_srt_path = ""
        
        try:
            # 1. 提取音频
            if not args.translate_only and not args.embed_only:
                print("\n步骤1: 提取音频")
                audio_path = extract_audio(args.input)
                if not audio_path:
                    print("音频提取失败")
                    return 1
                    
                if args.extract_only:
                    if args.audio:
                        import shutil
                        shutil.copy2(audio_path, args.audio)
                        print(f"音频已保存到: {args.audio}")
                    return 0
                    
            # 2. 转录字幕
            if not args.translate_only and not args.embed_only:
                print("\n步骤2: 转录字幕")
                segments, full_text = transcribe_audio(audio_path)
                if not segments:
                    print("语音识别失败")
                    return 1
                    
                # 创建字幕文件
                if not create_subtitle_file(segments, args.srt):
                    print("字幕文件创建失败")
                    return 1
                    
                if args.transcribe_only:
                    print(f"字幕已保存到: {args.srt}")
                    return 0
                    
            # 3. 翻译字幕
            if not args.embed_only:
                print("\n步骤3: 翻译字幕")
                srt_path = args.srt
                translated_srt_path = translate_subtitle_file(
                    srt_path,
                    target_lang=args.target_lang,
                    use_ollama=args.use_ollama,
                    model_name=args.model_name,
                    retry_translate=args.retry_translate
                )
                
                if not translated_srt_path:
                    print("字幕翻译失败")
                    return 1
                    
                if args.translate_only:
                    print(f"翻译后的字幕已保存到: {translated_srt_path}")
                    return 0
                    
            # 4. 嵌入字幕
            print("\n步骤4: 嵌入字幕")
            if not translated_srt_path and args.embed_only:
                translated_srt_path = args.srt
                
            options = {
                'hard_sub': args.hard_sub,
                'format': args.format,
                'subtitle_style': args.subtitle_style
            }
            
            if not embed_subtitles(args.input, translated_srt_path, args.output, options):
                print("字幕嵌入失败")
                return 1
                
            print(f"处理完成，输出文件: {args.output}")
            return 0
            
        finally:
            # 清理临时文件
            temp_manager.cleanup()
            
    except KeyboardInterrupt:
        print("\n用户中断处理")
        temp_manager.cleanup()
        return 1
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        temp_manager.cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main())