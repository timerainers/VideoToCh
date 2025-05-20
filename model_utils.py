import os
import sys
import platform
import subprocess
from typing import List, Dict, Tuple

def check_ffmpeg() -> bool:
    """检查系统是否安装了FFmpeg"""
    try:
        subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
        return True
    except:
        return False

def check_cuda() -> Tuple[bool, str]:
    """检查CUDA是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version}"
        return False, "CUDA未安装或不可用"
    except ImportError:
        return False, "PyTorch未安装"
    except Exception as e:
        return False, f"检查CUDA时出错: {str(e)}"

def install_package(package_name: str, version: str = None) -> Tuple[bool, str]:
    """安装指定的Python包
    
    Args:
        package_name: 包名
        version: 版本号（可选）
    
    Returns:
        (bool, str): (是否成功, 消息)
    """
    try:
        cmd = [sys.executable, '-m', 'pip', 'install']
        if version:
            cmd.append(f'{package_name}=={version}')
        else:
            cmd.append(package_name)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True, f"{package_name} 安装成功"
        else:
            return False, f"安装失败: {result.stderr}"
    except Exception as e:
        return False, f"安装出错: {str(e)}"

def check_whisper_installation() -> Tuple[bool, str]:
    """检查Whisper是否已安装"""
    try:
        import whisper
        return True, f"Whisper已安装 (版本: {whisper.__version__})"
    except ImportError:
        return False, "Whisper未安装"
    except Exception as e:
        return False, f"检查Whisper时出错: {str(e)}"

def get_whisper_models() -> List[Dict[str, str]]:
    """获取已安装的Whisper模型信息"""
    models_info = []
    model_sizes = {
        "tiny": "约40MB",
        "base": "约140MB",
        "small": "约460MB",
        "medium": "约1.5GB",
        "large": "约3GB"
    }
    
    try:
        import whisper
        
        # 获取缓存目录
        if platform.system() == "Windows":
            cache_dir = os.path.expandvars("%USERPROFILE%\\.cache\\whisper")
        else:
            cache_dir = os.path.expanduser("~/.cache/whisper")
            
        # 检查每个已知模型
        for model_name, default_size in model_sizes.items():
            model_info = {
                "name": model_name,
                "status": "未下载",
                "size": default_size,
                "path": ""
            }
            
            model_file = os.path.join(cache_dir, f"{model_name}.pt")
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                model_info.update({
                    "status": "已下载",
                    "size": f"{size_mb:.1f}MB",
                    "path": model_file
                })
            
            models_info.append(model_info)
            
    except Exception as e:
        print(f"获取模型信息时出错: {e}")
    
    return models_info

def download_whisper_model(model_name: str, progress_callback=None) -> Tuple[bool, str]:
    """下载指定的Whisper模型
    
    Args:
        model_name: 模型名称 (tiny/base/small/medium/large)
        progress_callback: 进度回调函数，接收参数：(进度值, 状态消息)
    
    Returns:
        (bool, str): (是否成功, 消息)
    """
    try:
        import whisper
        import torch
        
        # 检查模型名称是否有效
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if model_name not in valid_models:
            return False, f"无效的模型名称: {model_name}"
        
        if progress_callback:
            progress_callback(0, f"开始下载 {model_name} 模型...")
        
        try:
            # 设置torch.load的安全选项
            torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
            
            # 下载并加载模型
            model = whisper.load_model(model_name)
            
            if progress_callback:
                progress_callback(100, f"{model_name} 模型下载完成")
            
            return True, f"{model_name} 模型下载成功"
            
        except Exception as e:
            error_msg = f"下载模型失败: {str(e)}"
            if "weights_only=False" in str(e):
                # 处理torch.load的警告
                error_msg = "模型下载成功，但加载时出现警告。这不影响使用。"
                return True, error_msg
            raise e
            
    except Exception as e:
        error_msg = f"下载模型失败: {str(e)}"
        if progress_callback:
            progress_callback(0, error_msg)
        return False, error_msg

def check_environment() -> List[Dict[str, str]]:
    """检查环境配置状态"""
    status = []
    
    # 检查Python版本
    status.append({
        "name": "Python",
        "status": "已安装",
        "version": platform.python_version(),
        "details": sys.executable
    })
    
    # 检查CUDA
    cuda_available, cuda_info = check_cuda()
    status.append({
        "name": "CUDA",
        "status": "可用" if cuda_available else "不可用",
        "version": cuda_info if cuda_available else "",
        "details": cuda_info
    })
    
    # 检查FFmpeg
    ffmpeg_installed = check_ffmpeg()
    status.append({
        "name": "FFmpeg",
        "status": "已安装" if ffmpeg_installed else "未安装",
        "version": "",
        "details": "用于音频处理"
    })
    
    # 检查Whisper
    whisper_installed, whisper_info = check_whisper_installation()
    status.append({
        "name": "Whisper",
        "status": "已安装" if whisper_installed else "未安装",
        "version": whisper_info if whisper_installed else "",
        "details": "语音识别模型"
    })
    
    return status

def setup_environment() -> Tuple[bool, List[str]]:
    """安装和配置所需的环境
    
    Returns:
        (bool, List[str]): (是否全部成功, 操作日志)
    """
    logs = []
    success = True
    
    # 安装必要的包
    packages = {
        "torch": "2.2.1",
        "openai-whisper": "20231117",
        "PyQt6": None,  # 最新版本
        "pysrt": "1.1.2",
        "tqdm": "4.66.5",
        "requests": "2.31.0",
        "ffmpeg-python": "0.2.0",
        "moviepy": "1.0.3"
    }
    
    for package, version in packages.items():
        success, msg = install_package(package, version)
        logs.append(msg)
        if not success:
            success = False
    
    # 检查FFmpeg
    if not check_ffmpeg():
        logs.append("警告: 未检测到FFmpeg，请手动安装: https://ffmpeg.org/download.html")
        success = False
    
    return success, logs 