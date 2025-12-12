import os
import re
import torch
import numpy as np
from pathlib import Path
import json
import hashlib
import subprocess
import tempfile
import sys

# --------------------------------------------------------------------------
# 辅助函数：获取ComfyUI根目录
# --------------------------------------------------------------------------
def get_comfyui_root():
    """获取ComfyUI的根目录"""
    # 尝试多种方法找到ComfyUI根目录
    
    # 方法1: 通过当前文件路径向上查找
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        # 查找包含main.py或comfy目录的父目录
        if (parent / "main.py").exists() or (parent / "comfy").exists():
            return parent
    
    # 方法2: 使用sys.path中的路径
    for path in sys.path:
        path_obj = Path(path)
        if (path_obj / "main.py").exists() or (path_obj / "comfy").exists():
            return path_obj
    
    # 方法3: 使用当前工作目录
    cwd = Path.cwd()
    if (cwd / "main.py").exists() or (cwd / "comfy").exists():
        return cwd
    
    # 如果都找不到，返回当前工作目录
    return cwd

def resolve_path(path_str):
    """解析路径，相对路径基于ComfyUI根目录"""
    if not path_str or not path_str.strip():
        return ""
    
    path_str = path_str.strip()
    path_obj = Path(path_str)
    
    # 如果是绝对路径，直接返回
    if path_obj.is_absolute():
        return str(path_obj)
    
    # 如果是相对路径，基于ComfyUI根目录
    comfyui_root = get_comfyui_root()
    resolved_path = comfyui_root / path_str
    return str(resolved_path)

def move_to_recycle_bin(file_path):
    """将文件移动到回收站
    
    Args:
        file_path: 要移动的文件路径
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 使用 send2trash 库（跨平台支持）
        try:
            from send2trash import send2trash
            send2trash(str(file_path))
            return True
        except ImportError:
            # 如果没有安装 send2trash，使用系统特定的方法
            import platform
            system = platform.system()
            
            if system == 'Windows':
                # Windows: 使用 PowerShell 移动到回收站
                import subprocess
                ps_command = f'''
                Add-Type -AssemblyName Microsoft.VisualBasic
                [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
                    '{str(file_path)}',
                    'OnlyErrorDialogs',
                    'SendToRecycleBin'
                )
                '''
                result = subprocess.run(
                    ['powershell', '-Command', ps_command],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            
            elif system == 'Darwin':  # macOS
                # macOS: 使用 osascript 移动到废纸篓
                import subprocess
                result = subprocess.run(
                    ['osascript', '-e', f'tell application "Finder" to delete POSIX file "{str(file_path)}"'],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            
            elif system == 'Linux':
                # Linux: 使用 gio trash 或 gvfs-trash
                import subprocess
                # 尝试 gio trash
                result = subprocess.run(
                    ['gio', 'trash', str(file_path)],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    return True
                
                # 尝试 gvfs-trash
                result = subprocess.run(
                    ['gvfs-trash', str(file_path)],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            
            else:
                # 不支持的系统，返回False
                return False
                
    except Exception as e:
        print(f"  移动到回收站失败: {e}")
        return False

# --------------------------------------------------------------------------
# 视频帧加载函数 (使用 FFmpeg，参考 VideoHelperSuite 实现)
# --------------------------------------------------------------------------
def load_video_frames(video_path, scale_short_side=0):
    """使用 FFmpeg 加载视频并提取所有帧、音频和元数据
    
    Args:
        video_path: 视频文件路径
        scale_short_side: 按短边缩放的目标尺寸，0表示自动规范化为16的倍数
    """
    import time
    total_start = time.time()
    
    try:
        # 获取视频信息
        info_start = time.time()
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames,duration,width,height',
            '-of', 'json',
            str(video_path)
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(probe_result.stdout)
        
        info_time = time.time() - info_start
        print(f"  [加载] 获取视频信息耗时: {info_time:.2f}秒")
        
        if not video_info.get('streams'):
            print(f"无法获取视频流信息: {video_path}")
            return None, None, 0, 0
        
        stream = video_info['streams'][0]
        
        # 解析帧率
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
        
        # 检查是否是 NTSC 帧率（29.97）
        if abs(fps - 29.97) < 0.01:
            print(f"  [加载] 检测到 NTSC 帧率: {fps:.6f} fps (r_frame_rate: {fps_str})")
        elif abs(fps - 30.0) < 0.01:
            print(f"  [加载] 检测到标准 30 fps (r_frame_rate: {fps_str})")
        else:
            print(f"  [加载] 帧率: {fps:.6f} fps (r_frame_rate: {fps_str})")
        
        # 获取视频尺寸
        orig_width = int(stream.get('width', 0))
        orig_height = int(stream.get('height', 0))
        
        # 计算缩放后的尺寸（按短边缩放，保持宽高比）
        # 如果scale_short_side为0，自动规范化为16的倍数
        if scale_short_side > 0:
            # 缩放模式
            short_side = min(orig_width, orig_height)
            long_side = max(orig_width, orig_height)
            scale_ratio = scale_short_side / short_side
            
            if orig_width < orig_height:
                # 宽度是短边
                width = scale_short_side
                height = int(orig_height * scale_ratio)
            else:
                # 高度是短边
                height = scale_short_side
                width = int(orig_width * scale_ratio)
            
            # 确保尺寸是16的倍数（视频编码最佳性能）
            width = width - (width % 16)
            height = height - (height % 16)
            
            print(f"  [加载] 原始尺寸: {orig_width}x{orig_height}")
            print(f"  [加载] 缩放后尺寸: {width}x{height} (短边: {scale_short_side})")
        else:
            # 规范化模式：自动调整为16的倍数
            width = orig_width - (orig_width % 16)
            height = orig_height - (orig_height % 16)
            
            if width != orig_width or height != orig_height:
                print(f"  [加载] 原始尺寸: {orig_width}x{orig_height}")
                print(f"  [加载] 规范化尺寸: {width}x{height} (自动调整为16的倍数)")
            else:
                print(f"  [加载] 尺寸: {orig_width}x{orig_height} (已是16的倍数)")
        
        # 获取总帧数
        total_frames = int(stream.get('nb_frames', 0))
        if total_frames == 0:
            duration = float(stream.get('duration', 0))
            if duration > 0:
                total_frames = int(duration * fps)
        
        # 使用 FFmpeg 提取所有帧为原始数据
        decode_start = time.time()
        video_cmd = [
            'ffmpeg',
            '-i', str(video_path),
        ]
        
        # 如果需要缩放，添加缩放滤镜
        if scale_short_side > 0:
            video_cmd.extend(['-vf', f'scale={width}:{height}'])
        
        video_cmd.extend([
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ])
        
        print(f"  [加载] 开始解码视频帧...")
        # 使用 Popen 读取数据流
        process = subprocess.Popen(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        video_data, stderr = process.communicate()
        
        decode_time = time.time() - decode_start
        print(f"  [加载] 解码视频帧耗时: {decode_time:.2f}秒，数据大小: {len(video_data) / 1024 / 1024:.2f} MB")
        
        if process.returncode != 0:
            print(f"FFmpeg 提取帧失败: {stderr.decode('utf-8', errors='ignore')[:200]}")
            return None, None, 0, 0
        
        # 转换为 numpy 数组
        convert_start = time.time()
        frame_data = np.frombuffer(video_data, dtype=np.uint8)
        
        # 计算实际帧数
        bytes_per_frame = width * height * 3
        actual_frames = len(frame_data) // bytes_per_frame
        
        if actual_frames == 0:
            print(f"未能提取任何帧: {video_path}")
            return None, None, 0, 0
        
        # Reshape 为 (frames, height, width, 3)
        frame_data = frame_data[:actual_frames * bytes_per_frame]
        frames = frame_data.reshape((actual_frames, height, width, 3))
        frames = frames.astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames)
        
        convert_time = time.time() - convert_start
        print(f"  [加载] 转换为tensor耗时: {convert_time:.2f}秒")
        
        # 提取音频（参考 VideoHelperSuite 的实现）
        audio_start = time.time()
        audio_dict = None
        try:
            audio_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',
                '-f', 'f32le',
                '-'
            ]
            
            # 使用 Popen 读取音频数据流
            audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_data_bytes, audio_stderr = audio_process.communicate()
            
            if audio_process.returncode != 0:
                raise subprocess.CalledProcessError(audio_process.returncode, audio_cmd)
            
            # 解析音频信息
            import re
            match = re.search(r', (\d+) Hz, (\w+), ', audio_stderr.decode('utf-8', errors='ignore'))
            
            if match:
                sample_rate = int(match.group(1))
                channels_str = match.group(2)
                
                # 解析声道数
                if 'stereo' in channels_str:
                    channels = 2
                elif 'mono' in channels_str:
                    channels = 1
                else:
                    channels = 2  # 默认立体声
            else:
                sample_rate = 44100
                channels = 2
            
            # 转换音频数据
            # 检查音频数据是否为空
            if len(audio_data_bytes) > 0:
                audio_data = torch.frombuffer(bytearray(audio_data_bytes), dtype=torch.float32)
                
                if len(audio_data) > 0:
                    # Reshape 为 (samples, channels) 然后转置为 (channels, samples)
                    audio_data = audio_data.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
                    audio_dict = {'waveform': audio_data, 'sample_rate': sample_rate}
                    audio_time = time.time() - audio_start
                    print(f"  [加载] 提取音频耗时: {audio_time:.2f}秒")
                else:
                    print(f"  [加载] 视频无音频数据，使用空音频")
                    audio_dict = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
            else:
                print(f"  [加载] 视频无音频数据，使用空音频")
                audio_dict = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
                
        except subprocess.CalledProcessError:
            print(f"视频无音频轨道或音频提取失败: {video_path}")
            audio_dict = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
        
        total_time = time.time() - total_start
        print(f"  [加载] 总耗时: {total_time:.2f}秒")
        
        return frames_tensor, audio_dict, fps, actual_frames
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 处理失败: {str(e)}")
        return None, None, 0, 0
    except Exception as e:
        print(f"视频加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0

# --------------------------------------------------------------------------
# 视频目录加载器节点
# --------------------------------------------------------------------------
class VideoDirectoryLoader:
    _auto_index = {}  # 自动索引库（持久化到文件）
    
    @classmethod
    def _get_cache_file(cls):
        """获取缓存文件路径"""
        cache_dir = Path(__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "video_auto_index.json"
    
    @classmethod
    def _load_auto_index(cls):
        """从文件加载自动索引"""
        cache_file = cls._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cls._auto_index = json.load(f)
            except Exception as e:
                print(f"加载自动索引缓存失败: {e}")
                cls._auto_index = {}
        else:
            cls._auto_index = {}
    
    @classmethod
    def _save_auto_index(cls):
        """保存自动索引到文件"""
        try:
            cache_file = cls._get_cache_file()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cls._auto_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存自动索引缓存失败: {e}")
    
    @classmethod
    def _get_key(cls, 目录路径, 任务批次编号):
        """生成缓存键"""
        key_str = f"{目录路径}#{任务批次编号}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "目录路径": ("STRING", {"default": "output/长视频转换分割", "multiline": False, "placeholder": "支持目录或单个视频文件路径", "tooltip": "视频目录或单个视频文件的路径。支持目录（批量加载视频）和单个文件两种模式。支持相对路径（基于ComfyUI根目录）和绝对路径。"}),
                "起始索引": ("INT", {"default": 0, "min": 0, "step": 1, "description": "从第几个视频开始（0表示第1个）", "tooltip": "视频加载的起始位置（0表示第1个视频）。配合单视频顺序加载模式实现批量处理，每次运行自动加载下一个视频。修改此值可指定从哪个视频开始。"}),
                "任务批次编号": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1, "description": "任务批次标识，变化时重置自动索引", "tooltip": "任务批次标识符。不同批次编号将重置自动索引缓存，便于区分不同处理任务。同一批次编号会使用共享的自动索引记录。"}),
                "sort_method": (["按名称", "按数字", "按修改时间"], {"default": "按数字", "description": "视频排序方式", "tooltip": "视频排序方式。按名称：按文件名字母顺序；按数字：按文件名中的数字排序（适合序列文件）；按修改时间：按文件最后修改时间排序。"}),
                "递归搜索子目录": ("BOOLEAN", {"default": False, "description": "是否递归查找所有子文件夹", "tooltip": "是否递归搜索所有子目录中的视频文件。开启后将在目录及其所有子目录中查找视频文件；关闭则仅搜索当前目录。"}),
                "文件扩展名过滤": ("STRING", {"default": "", "placeholder": "用逗号分隔，如: mp4,avi,mkv", "description": "留空则加载所有支持的视频格式", "tooltip": "文件扩展名过滤器。用逗号分隔多个扩展名，如'mp4,avi,mkv'。留空则加载所有支持的视频格式。帮助限制仅加载指定格式的视频文件。"}),
                "加载失败跳过": ("BOOLEAN", {"default": True, "description": "加载失败时是否跳过", "tooltip": "当视频加载失败时是否自动跳过。开启后跳过损坏或不支持的文件继续处理；关闭后遇到加载失败将停止处理。建议开启以确保批量处理稳定性。"}),
                "按短边缩放": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16, "description": "按短边缩放，0表示自动规范化为16的倍数（推荐）", "tooltip": "视频缩放尺寸（按短边像素数）。设置为0时自动规范化为16的倍数（推荐）；其他值则按指定短边长度缩放。有助于优化处理性能和内存使用。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("帧序列", "音频", "帧率", "总帧数", "可用总数", "剩余未处理", "宽度", "高度")
    FUNCTION = "load_video"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "从指定目录或单个视频文件加载视频并拆分为帧序列，支持音频提取、递归搜索、排序等功能。单视频顺序加载模式配合队列实现批量处理。所有视频处理完成后会自动跳过执行。"

    def load_video(self, 目录路径, 起始索引, 任务批次编号, sort_method, 递归搜索子目录, 文件扩展名过滤, 加载失败跳过, 按短边缩放):
        """
        从指定目录或单个视频文件加载视频并拆分为帧序列，支持音频提取、递归搜索、排序等功能
        
        Args:
            目录路径 (str): 视频目录路径或单个视频文件路径，支持相对路径和绝对路径
            起始索引 (int): 从第几个视频开始（0表示第1个）
            任务批次编号 (int): 任务批次标识符，用于区分不同的加载任务
            sort_method (str): 排序方式，可选值："按名称"、"按数字"、"按修改时间"
            递归搜索子目录 (bool): 是否递归搜索所有子目录
            文件扩展名过滤 (str): 用逗号分隔的文件扩展名列表
            加载失败跳过 (bool): 加载失败时是否自动跳过
            按短边缩放 (int): 按短边缩放的目标尺寸，0表示自动规范化为16的倍数
        
        Returns:
            tuple: 包含以下元素的元组
                - 帧序列 (torch.Tensor): 加载的视频帧张量，形状为(B, H, W, C)
                - 音频 (dict): 音频数据字典，包含waveform和sample_rate
                - 帧率 (float): 视频帧率
                - 总帧数 (int): 视频的总帧数
                - 可用总数 (int): 目录中可用视频总数
                - 剩余未处理 (int): 剩余待处理视频数量
                - 宽度 (int): 视频帧宽度
                - 高度 (int): 视频帧高度
        
        Notes:
            - 支持单个视频文件和目录批量加载
            - 单视频顺序加载模式使用持久化缓存，确保每次运行自动加载下一个视频
            - 所有视频处理完成后会自动跳过执行
            - 支持中文路径和多种视频格式
            - 自动检测并跳过无效视频文件
            - 音频数据格式：{'waveform': torch.Tensor, 'sample_rate': int}
        """
        # 后台默认开启的选项
        单视频顺序加载 = True  # 默认开启单视频顺序加载模式
        智能队列建议 = True  # 默认开启智能队列建议
        import time
        start_time = time.time()
        
        empty_audio = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
        
        # 解析路径（相对路径基于ComfyUI根目录）
        目录路径 = resolve_path(目录路径)
        
        # 检查是文件还是目录
        path = Path(目录路径)
        
        # 支持所有 FFmpeg 支持的视频格式
        if 文件扩展名过滤.strip():
            video_extensions = tuple(f".{ext.strip().lower()}" for ext in 文件扩展名过滤.split(",") if ext.strip())
        else:
            # 常见视频格式
            video_extensions = (
                ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", 
                ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts", ".mts", 
                ".m2ts", ".vob", ".rm", ".rmvb", ".asf", ".divx"
            )
        
        # 判断是单个视频文件还是目录
        if path.is_file():
            # 单个视频文件模式
            if path.suffix.lower() in video_extensions:
                video_paths = [path]
                print(f"[模式] 单个视频文件: {path.name}")
            else:
                print(f"错误: 文件 '{目录路径}' 不是支持的视频格式")
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, 0, 0, 0, 0)
        elif path.is_dir():
            # 目录模式
            # 优化：使用集合避免重复，使用生成器提高性能
            video_paths_set = set()
            
            if 递归搜索子目录:
                # 递归搜索所有子目录
                for ext in video_extensions:
                    video_paths_set.update(path.rglob(f'*{ext}'))
            else:
                # 只搜索当前目录
                for ext in video_extensions:
                    video_paths_set.update(path.glob(f'*{ext}'))
            
            video_paths = list(video_paths_set)
        else:
            print(f"错误: 路径 '{目录路径}' 不存在")
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, 0, 0, 0, 0)
        
        total_available = len(video_paths)
        
        scan_time = time.time() - start_time
        print(f"[性能] 扫描目录耗时: {scan_time:.2f}秒，找到 {total_available} 个视频")
        
        if total_available == 0:
            print("未找到任何视频文件")
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, 0, 0, 0, 0)

        # 排序（优化性能）
        sort_start = time.time()
        if sort_method == "按名称":
            # 按文件名排序（最快）
            video_paths.sort(key=lambda x: x.name.lower())
        elif sort_method == "按数字":
            # 按文件名中的数字排序
            def numeric_sort_key(item):
                numbers = re.findall(r'\d+', item.name)
                return tuple(map(int, numbers)) if numbers else (float('inf'),)
            video_paths.sort(key=numeric_sort_key)
        elif sort_method == "按修改时间":
            # 按修改时间排序（较慢，需要读取文件元数据）
            try:
                video_paths.sort(key=lambda x: x.stat().st_mtime)
            except Exception as e:
                print(f"警告: 按修改时间排序失败，改用按名称排序: {e}")
                video_paths.sort(key=lambda x: x.name.lower())
        
        sort_time = time.time() - sort_start
        print(f"[性能] 排序耗时: {sort_time:.2f}秒 (方法: {sort_method})")

        # 确定起始索引
        if 单视频顺序加载:
            # 加载自动索引缓存
            VideoDirectoryLoader._load_auto_index()
            
            cache_key = VideoDirectoryLoader._get_key(目录路径, str(任务批次编号))
            
            # 检查配置变化
            last_config_key = "_last_config"
            current_config = f"{目录路径}#{任务批次编号}"
            
            if (last_config_key not in VideoDirectoryLoader._auto_index or 
                VideoDirectoryLoader._auto_index[last_config_key] != current_config):
                VideoDirectoryLoader._auto_index[cache_key] = 起始索引
                VideoDirectoryLoader._auto_index[last_config_key] = current_config
                VideoDirectoryLoader._save_auto_index()
                print(f"检测到配置变化，已重置自动索引到起始位置: {起始索引}")
            
            if cache_key not in VideoDirectoryLoader._auto_index:
                VideoDirectoryLoader._auto_index[cache_key] = 起始索引
                VideoDirectoryLoader._save_auto_index()
            
            # 获取当前索引并递增（为下次运行准备）
            start = VideoDirectoryLoader._auto_index[cache_key]
            next_index = start + 1
            
            # 立即保存递增后的索引（为下次运行准备）
            VideoDirectoryLoader._auto_index[cache_key] = next_index
            VideoDirectoryLoader._save_auto_index()
        else:
            start = 起始索引
        
        # 单视频模式只加载一个
        if start >= total_available:
            if 单视频顺序加载:
                print(f"✓ 所有视频已处理完成，跳过执行")
                print(f"  - 总视频数: {total_available}")
                print(f"  - 当前索引: {start}")
                print(f"  - 目录路径: {目录路径}")
                print(f"  - 任务批次: {任务批次编号}")
                print(f"💡 提示: 如需重新处理，请修改目录路径或任务批次编号")
                # 返回空数据，静默跳过
                remaining = 0
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, total_available, remaining, 0, 0)
            else:
                print(f"起始索引 {start} 超出范围，可用视频数: {total_available}")
                remaining = 0
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, total_available, remaining, 0, 0)
        
        # 直接使用索引（现在索引从 0 开始）
        video_path = video_paths[start]
        
        # 加载视频
        load_start = time.time()
        frames, audio, fps, total_frames = load_video_frames(str(video_path), 按短边缩放)
        load_time = time.time() - load_start
        print(f"[性能] 加载视频耗时: {load_time:.2f}秒")
        
        # 获取实际的宽度和高度
        if frames is not None and frames.shape[0] > 0:
            actual_height = frames.shape[1]
            actual_width = frames.shape[2]
        else:
            actual_width = 0
            actual_height = 0
        
        if frames is None:
            if 加载失败跳过:
                print(f"跳过加载失败的视频: {video_path}")
                # 注意：索引已经在前面递增了，这里不需要再次更新
                remaining = max(0, total_available - start - 1)
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, total_available, remaining, 0, 0)
            else:
                print(f"加载失败: {video_path}")
                remaining = max(0, total_available - start - 1)
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), 0.0, 0, total_available, remaining, 0, 0)
        
        # 更新索引（注意：索引已经在前面递增并保存了）
        if 单视频顺序加载:
            # 计算剩余未处理数量（不包括当前这个，因为当前这个正在处理）
            remaining = max(0, total_available - start - 1)
            
            # 智能提示（显示为 1-based 索引更友好）
            display_index = start + 1
            if 智能队列建议 and remaining > 0:
                print(f"当前加载: {video_path.name} (第 {display_index}/{total_available} 个)")
                print(f"  - 帧数: {total_frames}, 帧率: {fps:.2f} fps")
                print(f"  - 剩余未处理: {remaining} 个视频")
                print(f"💡 智能建议: 下次从索引 {start + 1} 开始，队列设置为 {remaining} 次可完成剩余视频处理")
            else:
                print(f"当前加载: {video_path.name} (第 {display_index}/{total_available} 个)")
                print(f"  - 帧数: {total_frames}, 帧率: {fps:.2f} fps")
                if remaining == 0:
                    print(f"✓ 这是最后一个视频")
        else:
            remaining = 0
        
        return (frames, audio, fps, total_frames, total_available, remaining, actual_width, actual_height)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 单视频顺序加载模式默认开启：返回 NaN 确保每次都执行
        # 原因：需要每次检查索引状态，判断是否已完成
        # 如果已完成，会在 load_video 方法中静默跳过
        return float("NaN")

# --------------------------------------------------------------------------
# 视频自动合并器节点
# --------------------------------------------------------------------------
class VideoAutoCombine:
    """自动保存视频，达到目标数量后合并"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "帧率": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.01, "tooltip": "输出视频的帧率（1-120 FPS）。控制视频播放流畅度。常用设置：电影24fps，视频30fps，慢动作60fps。帧率越高文件越大。"}),
                "触发合并数量": ("INT", {"default": 999, "min": 1, "max": 9999, "step": 1, "description": "达到此数量后自动合并", "tooltip": "自动触发合并的最小视频数量。当目录中的视频文件达到此数量时，将自动开始合并处理。可以避免频繁合并，实现批量高效处理。"}),
                "最终文件名": ("STRING", {"default": "merged_video", "multiline": False, "placeholder": "合并后的文件名（不含扩展名），可连接视频转换分割器的原文件名输出", "tooltip": "合并后视频的文件名（不含扩展名）。可直接连接视频转换分割器的原文件名输出端口，自动使用原始视频名称。扩展名将根据视频编码类型自动添加。"}),
                "待合并视频目录": ("STRING", {"default": "output/待合并视频", "multiline": False, "placeholder": "临时保存待合并视频的目录", "tooltip": "临时保存待合并视频的目录路径。视频片段会先保存到此目录，积累到指定数量后自动合并。支持相对路径（基于ComfyUI根目录）和绝对路径。"}),
                "最终合并视频目录": ("STRING", {"default": "output/最终合并视频", "multiline": False, "placeholder": "最终合并后视频的保存目录", "tooltip": "最终合并视频的保存目录路径。合并完成后，视频将保存到此目录。支持相对路径（基于ComfyUI根目录）和绝对路径。目录不存在时会自动创建。"}),
                "视频编码": (["h264", "h265", "vp9", "prores"], {"default": "h264", "tooltip": "视频编码格式。h264：通用兼容性好；h265：压缩效率更高，文件更小但兼容性稍差；vp9：Web优化格式；prores：专业编辑格式。推荐h264获得最佳兼容性。"}),
                "视频质量": ("INT", {"default": 23, "min": 0, "max": 51, "step": 1, "description": "CRF值，越小质量越高(h264/h265)", "tooltip": "视频编码质量控制（CRF值）。值越小质量越高，文件越大。推荐设置：高质量18-20，中等质量23-25，低质量28-30。0为无损（文件极大）。仅对h264/h265有效。"}),
                "独立运行模式": ("BOOLEAN", {"default": False, "description": "开启后直接合并目录中的视频（合并后不删除原文件）", "tooltip": "独立运行模式开关。开启后直接合并目录中现有的视频文件，不需要输入帧序列。合并后会将原始文件移动到回收站。适用于手动整理和合并现有视频。"}),
            },
            "optional": {
                "帧序列": ("IMAGE", {"tooltip": "输入的图像帧序列。常规模式下必需此输入，用于生成视频片段。支持批量输入多个帧序列，自动保存为临时视频文件等待合并。"}),
                "音频": ("AUDIO", {"tooltip": "可选的音频输入。连接后将作为背景音乐或音频轨道与视频合并。可以是单个音频文件或音频序列。"}),
                "原音频路径": ("STRING", {"default": "", "multiline": False, "placeholder": "连接视频转换分割器的原音频输出，用于最终合并", "forceInput": True, "tooltip": "原始视频的音频路径。连接视频转换分割器的原音频输出端口，用于在最终合并时保持原始音轨同步。确保音频与视频完美匹配。"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "combine_video"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "保存视频到待合并目录，达到目标数量后自动合并并保存到最终合并视频目录。开启独立运行模式后，可直接合并目录中的视频文件（自动统一格式）。最终文件名可连接视频转换分割器的原文件名输出，扩展名将根据视频编码自动添加。"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 强制每次都重新执行，不使用缓存
        return float("NaN")

    def combine_video(self, 帧率, 触发合并数量, 最终文件名, 待合并视频目录, 最终合并视频目录, 视频编码, 视频质量, 独立运行模式, 帧序列=None, 音频=None, 原音频路径=""):
        try:
            # 根据视频编码确定文件扩展名
            codec_to_ext = {
                "h264": ".mp4",
                "h265": ".mp4",
                "vp9": ".webm",
                "prores": ".mov"
            }
            file_ext = codec_to_ext.get(视频编码, ".mp4")
            
            # 处理最终文件名：移除已有的扩展名（如果有），然后添加正确的扩展名
            import os
            final_name_without_ext = os.path.splitext(最终文件名)[0] if 最终文件名 else "merged_video"
            final_filename_with_ext = f"{final_name_without_ext}{file_ext}"
            
            print(f"\n{'='*60}")
            print(f"[视频合并器] 开始处理...")
            print(f"[视频合并器] 独立运行模式: {独立运行模式}")
            if not 独立运行模式 and 帧序列 is not None:
                print(f"[视频合并器] 帧序列形状: {帧序列.shape}")
                print(f"[视频合并器] 帧率: {帧率}")
            print(f"[视频合并器] 待合并视频目录: {待合并视频目录}")
            print(f"[视频合并器] 最终合并视频目录: {最终合并视频目录}")
            print(f"[视频合并器] 最终文件名: {final_filename_with_ext}")
            print(f"[视频合并器] 视频编码: {视频编码}")
            print(f"[视频合并器] 视频质量: {视频质量}")
            print(f"[视频合并器] 触发合并数量: {触发合并数量}")
            
            # 解析路径（相对路径基于ComfyUI根目录）
            待合并视频目录 = resolve_path(待合并视频目录)
            最终合并视频目录 = resolve_path(最终合并视频目录)
            print(f"[视频合并器] 解析后待合并路径: {待合并视频目录}")
            print(f"[视频合并器] 解析后最终路径: {最终合并视频目录}")
            
            # 处理保存目录
            temp_save_dir = Path(待合并视频目录)
            temp_save_dir.mkdir(parents=True, exist_ok=True)
            
            final_save_dir = Path(最终合并视频目录)
            final_save_dir.mkdir(parents=True, exist_ok=True)
            
            # 独立运行模式：直接合并目录中的视频
            if 独立运行模式:
                print(f"[视频合并器] 独立运行模式：检测待合并目录中的视频和音频...")
                
                # 获取所有视频文件（支持多种格式）
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm']
                all_videos = []
                for ext in video_extensions:
                    all_videos.extend(temp_save_dir.glob(ext))
                
                # 按文件名排序
                all_videos = sorted(all_videos)
                current_count = len(all_videos)
                
                # 检测音频文件（不计入触发合并数量）
                audio_extensions = ['*.mp3', '*.wav', '*.aac', '*.flac', '*.m4a', '*.ogg', '*.wma']
                all_audio_files = []
                for ext in audio_extensions:
                    all_audio_files.extend(temp_save_dir.glob(ext))
                
                # 按文件名排序
                all_audio_files = sorted(all_audio_files)
                audio_count = len(all_audio_files)
                
                print(f"[视频合并器] 找到 {current_count} 个视频文件")
                if audio_count > 0:
                    print(f"[视频合并器] 找到 {audio_count} 个音频文件（不计入触发数量）")
                    for i, a in enumerate(all_audio_files, 1):
                        file_size = a.stat().st_size / 1024 / 1024
                        print(f"  音频 {i}. {a.name} ({file_size:.2f} MB)")
                
                if current_count == 0:
                    print(f"[视频合并器] ⚠️ 待合并目录为空，无需合并")
                    print(f"{'='*60}\n")
                    return ()
                
                if current_count < 触发合并数量:
                    print(f"[视频合并器] 当前 {current_count} 个视频，未达到触发数量 {触发合并数量}")
                    print(f"[视频合并器] 还需要 {触发合并数量 - current_count} 个视频")
                    print(f"{'='*60}\n")
                    return ()
                
                print(f"[视频合并器] ★ 已达到触发数量 {触发合并数量}，开始合并...")
                
                # 显示待合并视频列表
                for i, v in enumerate(all_videos, 1):
                    file_size = v.stat().st_size / 1024 / 1024
                    print(f"  视频 {i}. {v.name} ({file_size:.2f} MB)")
                
                # 最终输出路径
                final_output = self._get_unique_filename(final_save_dir, final_filename_with_ext)
                
                # 合并视频（智能选择：格式一致用流复制，否则重新编码）
                video_list = [str(v) for v in all_videos]
                audio_file = str(all_audio_files[0]) if audio_count > 0 else None
                
                # 检查视频格式是否一致
                format_consistent = self._check_videos_format_consistency(video_list)
                
                if audio_file:
                    # 有音频文件：使用重新编码以确保音画同步
                    print(f"[视频合并器] 将使用音频文件: {Path(audio_file).name}")
                    print(f"[视频合并器] 模式: 重新编码（确保音画同步）")
                    self._merge_videos_with_normalization_and_audio(video_list, str(final_output), 视频编码, 视频质量, audio_file)
                elif format_consistent:
                    # 格式一致且无音频：使用流复制（快速）
                    print(f"[视频合并器] 模式: 流复制（快速合并，无需重新编码）")
                    self._merge_all_videos(video_list, str(final_output))
                else:
                    # 格式不一致：重新编码统一格式
                    print(f"[视频合并器] 模式: 重新编码（统一格式）")
                    self._merge_videos_with_normalization(video_list, str(final_output), 视频编码, 视频质量)
                
                # 独立运行模式：移动原文件到回收站
                print(f"[视频合并器] 独立运行模式：将原始视频文件移动到回收站...")
                success_count = 0
                fail_count = 0
                for video_file in all_videos:
                    if move_to_recycle_bin(video_file):
                        success_count += 1
                        print(f"  - 已移至回收站: {video_file.name}")
                    else:
                        fail_count += 1
                        print(f"  - 移动失败（保留）: {video_file.name}")
                
                if success_count > 0:
                    print(f"[视频合并器] ✓ 已将 {success_count} 个文件移至回收站")
                if fail_count > 0:
                    print(f"[视频合并器] ⚠️ {fail_count} 个文件移动失败（已保留）")
                    print(f"[视频合并器] 提示：如需清理，请手动删除待合并目录中的文件")
                
                print(f"[视频合并器] ✓ 合并完成!")
                print(f"[视频合并器] 最终视频路径: {final_output}")
                print(f"[视频合并器] 最终文件大小: {Path(final_output).stat().st_size / 1024 / 1024:.2f} MB")
                print(f"{'='*60}\n")
                return ()
            
            # 常规模式：保存帧序列并合并
            if 帧序列 is None:
                print(f"[视频合并器] ✗ 错误: 常规模式需要输入帧序列")
                print(f"{'='*60}\n")
                return ()
            
            # 获取待合并目录中现有的视频文件（按名称排序）
            existing_videos = sorted(temp_save_dir.glob("video_*.mp4"))
            current_count = len(existing_videos)
            
            print(f"[视频合并器] 待合并目录已有 {current_count} 个视频文件")
            
            # 生成新的视频文件名（固定使用 .mp4）
            new_video_index = current_count
            new_video_path = temp_save_dir / f"video_{new_video_index:04d}.mp4"
            
            # 检查帧序列是否有效
            if 帧序列.shape[0] == 0 or 帧序列.shape[1] <= 1 or 帧序列.shape[2] <= 1:
                print(f"[视频合并器] ✗ 跳过无效的帧序列: {帧序列.shape}")
                print(f"[视频合并器] 提示: 视频加载器可能没有成功加载视频")
                print(f"{'='*60}\n")
                return ()
            
            # 保存当前视频
            print(f"[视频合并器] 正在保存视频 {new_video_index + 1}/{触发合并数量}: {new_video_path.name}")
            self._save_frames_to_video(帧序列, str(new_video_path), 帧率, 视频编码, 视频质量, 音频)
            print(f"[视频合并器] ✓ 视频已保存: {new_video_path.name}")
            
            # 更新视频列表
            existing_videos.append(new_video_path)
            current_count += 1
            
            print(f"[视频合并器] 当前进度: {current_count}/{触发合并数量}")
            
            # 检查是否达到触发合并数量
            if current_count >= 触发合并数量:
                print(f"\n[视频合并器] ★ 已达到触发合并数量 {触发合并数量}，开始合并所有视频...")
                
                # 重新获取所有待合并视频文件（确保顺序正确）
                all_videos = sorted(temp_save_dir.glob("video_*.mp4"))
                print(f"[视频合并器] 找到 {len(all_videos)} 个待合并视频文件:")
                for v in all_videos:
                    print(f"  - {v.name}")
                
                # 最终输出路径（保存到最终合并视频目录）
                # 检查文件是否存在，如果存在则添加序号
                final_output = final_save_dir / final_filename_with_ext
                
                if final_output.exists():
                    # 文件已存在，生成带序号的新文件名
                    base_name = final_output.stem  # 不含扩展名的文件名
                    extension = final_output.suffix  # 扩展名
                    counter = 1
                    
                    while True:
                        new_filename = f"{base_name}_{counter:03d}{extension}"
                        final_output = final_save_dir / new_filename
                        if not final_output.exists():
                            break
                        counter += 1
                        if counter > 999:
                            print(f"[视频合并器] ⚠️ 警告: 已存在超过999个同名文件，使用时间戳")
                            import time
                            timestamp = int(time.time())
                            new_filename = f"{base_name}_{timestamp}{extension}"
                            final_output = final_save_dir / new_filename
                            break
                    
                    print(f"[视频合并器] 检测到同名文件，自动重命名为: {final_output.name}")
                
                if len(all_videos) == 1:
                    # 只有一个视频，直接复制到最终目录
                    import shutil
                    shutil.copy2(str(all_videos[0]), str(final_output))
                    print(f"[视频合并器] 单个视频，已复制到最终目录: {final_output.name}")
                else:
                    # 多个视频，直接使用流复制（常规模式下所有视频都是同一节点生成，格式必然一致）
                    video_list = [str(v) for v in all_videos]
                    print(f"[视频合并器] 模式: 流复制（常规模式，格式统一）")
                    self._merge_all_videos(video_list, str(final_output))
                
                # 如果有原音频输入，合并原音频到最终视频
                if 原音频路径 and os.path.exists(原音频路径):
                    print(f"[视频合并器] 检测到原音频输入，合并原音频到最终视频...")
                    print(f"[视频合并器] 原音频: {os.path.basename(原音频路径)}")
                    
                    try:
                        # 创建临时文件
                        temp_output = str(final_output) + '.tmp.mp4'
                        
                        # 合并原音频到视频
                        merge_audio_cmd = [
                            'ffmpeg',
                            '-y',
                            '-i', str(final_output),      # 输入视频
                            '-i', 原音频路径,              # 输入音频
                            '-c:v', 'copy',               # 视频流复制
                            '-c:a', 'aac',                # 音频编码为AAC
                            '-b:a', '192k',               # 音频比特率
                            '-map', '0:v:0',              # 使用第一个输入的视频流
                            '-map', '1:a:0',              # 使用第二个输入的音频流
                            '-shortest',                  # 以最短流为准
                            temp_output
                        ]
                        
                        result = subprocess.run(merge_audio_cmd, capture_output=True)
                        
                        if result.returncode == 0 and os.path.exists(temp_output):
                            # 替换原文件
                            os.replace(temp_output, str(final_output))
                            print(f"[视频合并器] ✓ 原音频合并成功")
                        else:
                            error_msg = result.stderr.decode('utf-8', errors='ignore')
                            print(f"[视频合并器] ⚠️ 原音频合并失败: {error_msg[:200]}")
                            # 清理临时文件
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                    except Exception as e:
                        print(f"[视频合并器] ⚠️ 原音频合并失败: {str(e)}")
                
                # 将待合并目录中的原始视频文件移动到回收站
                print(f"[视频合并器] 将临时视频文件移动到回收站...")
                success_count = 0
                fail_count = 0
                for video_file in all_videos:
                    if move_to_recycle_bin(video_file):
                        success_count += 1
                        print(f"  - 已移至回收站: {video_file.name}")
                    else:
                        fail_count += 1
                        print(f"  - 移动失败（保留）: {video_file.name}")
                
                if success_count > 0:
                    print(f"[视频合并器] ✓ 已将 {success_count} 个文件移至回收站")
                if fail_count > 0:
                    print(f"[视频合并器] ⚠️ {fail_count} 个文件移动失败（已保留）")
                    print(f"[视频合并器] 提示：可以从回收站恢复文件，或手动清理")
                
                print(f"[视频合并器] ✓ 合并完成!")
                print(f"[视频合并器] 最终视频路径: {final_output}")
                print(f"[视频合并器] 最终文件大小: {final_output.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print(f"[视频合并器] 等待更多视频...")
                print(f"[视频合并器] 还需要 {触发合并数量 - current_count} 个视频")
            
            print(f"{'='*60}\n")
            return ()
            
        except Exception as e:
            print(f"[视频合并器] ✗ 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return ()
    
    def _save_frames_to_video(self, frames, output_path, fps, codec, quality, audio=None):
        """将帧序列保存为视频文件"""
        print(f"  [编码] 开始编码视频...")
        
        # 转换帧数据
        if frames.ndim == 4:
            frames_np = frames.cpu().numpy()
        else:
            frames_np = frames.unsqueeze(0).cpu().numpy()
        
        # 转换为 uint8
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
        
        height, width = frames_np.shape[1:3]
        num_frames = frames_np.shape[0]
        
        print(f"  [编码] 视频尺寸: {width}x{height}, 帧数: {num_frames}")
        
        # 检测 GPU 支持
        use_gpu = self._check_nvidia_gpu() if hasattr(self, '_check_nvidia_gpu') else False
        
        # 设置编码器参数
        if codec == "h264":
            if use_gpu:
                print(f"  [编码] 使用 NVIDIA GPU 加速 (h264_nvenc)")
                codec_name = "h264_nvenc"
                codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
            else:
                codec_name = "libx264"
                codec_params = ["-crf", str(quality), "-preset", "medium"]
        elif codec == "h265":
            if use_gpu:
                print(f"  [编码] 使用 NVIDIA GPU 加速 (hevc_nvenc)")
                codec_name = "hevc_nvenc"
                codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
            else:
                codec_name = "libx265"
                codec_params = ["-crf", str(quality), "-preset", "medium"]
        elif codec == "vp9":
            codec_name = "libvpx-vp9"
            codec_params = ["-crf", str(quality), "-b:v", "0"]
        elif codec == "prores":
            codec_name = "prores_ks"
            codec_params = ["-profile:v", "3"]
        else:
            if use_gpu:
                print(f"  [编码] 使用 NVIDIA GPU 加速 (h264_nvenc)")
                codec_name = "h264_nvenc"
                codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
            else:
                codec_name = "libx264"
                codec_params = ["-crf", str(quality), "-preset", "medium"]
        
        # 处理音频数据
        has_audio = False
        audio_file = None
        
        if audio is not None:
            try:
                print(f"  [编码] 处理音频数据...")
                # 处理音频数据
                if callable(audio):
                    audio = audio()
                
                if isinstance(audio, dict) and 'waveform' in audio:
                    waveform = audio['waveform']
                    sample_rate = audio.get('sample_rate', 44100)
                    
                    print(f"  [编码] 音频采样率: {sample_rate}, 形状: {waveform.shape}")
                    
                    # 检查音频是否为空
                    if waveform.numel() > 0:
                        # 准备音频数据
                        if waveform.ndim == 3:
                            waveform = waveform.squeeze(0)  # 移除 batch 维度
                        
                        # 转置为 (samples, channels)
                        if waveform.shape[0] < waveform.shape[1]:
                            waveform = waveform.transpose(0, 1)
                        
                        # 转换为 float32 numpy
                        audio_np = waveform.cpu().numpy().astype(np.float32)
                        
                        # 计算视频时长和音频时长
                        video_duration = num_frames / fps  # 视频时长（秒）
                        audio_duration = len(audio_np) / sample_rate  # 音频时长（秒）
                        
                        print(f"  [编码] 视频时长: {video_duration:.3f}秒, 音频时长: {audio_duration:.3f}秒")
                        
                        # 保存音频到临时文件
                        audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        
                        # 计算音频调整参数
                        # 如果音频时长与视频时长不匹配，使用 atempo 滤镜调整
                        tempo_ratio = audio_duration / video_duration if video_duration > 0 else 1.0
                        
                        # atempo 滤镜的范围是 0.5 到 2.0，如果超出需要链式调用
                        audio_filters = []
                        if abs(tempo_ratio - 1.0) > 0.01:  # 差异超过1%才调整
                            print(f"  [编码] 音频时长不匹配，调整速度比例: {tempo_ratio:.3f}")
                            
                            # 如果比例超出范围，需要分步调整
                            current_ratio = tempo_ratio
                            while current_ratio > 2.0:
                                audio_filters.append("atempo=2.0")
                                current_ratio /= 2.0
                            while current_ratio < 0.5:
                                audio_filters.append("atempo=0.5")
                                current_ratio /= 0.5
                            if abs(current_ratio - 1.0) > 0.01:
                                audio_filters.append(f"atempo={current_ratio:.6f}")
                        
                        # 使用 FFmpeg 将原始音频转换为 WAV，并调整时长
                        audio_cmd = [
                            'ffmpeg',
                            '-y',
                            '-f', 'f32le',
                            '-ar', str(sample_rate),
                            '-ac', str(audio_np.shape[1]),
                            '-i', '-',
                        ]
                        
                        # 添加音频滤镜
                        if audio_filters:
                            audio_cmd.extend(['-af', ','.join(audio_filters)])
                        
                        audio_cmd.extend([
                            '-c:a', 'pcm_s16le',
                            audio_file.name
                        ])
                        
                        audio_process = subprocess.Popen(
                            audio_cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        # 使用 communicate 避免管道阻塞
                        audio_data = audio_np.tobytes()
                        print(f"  [编码] 音频数据大小: {len(audio_data) / 1024 / 1024:.2f} MB")
                        
                        try:
                            stdout, stderr = audio_process.communicate(input=audio_data, timeout=60)
                        except subprocess.TimeoutExpired:
                            audio_process.kill()
                            stdout, stderr = audio_process.communicate()
                            raise Exception("音频转换超时")
                        
                        if audio_process.returncode == 0:
                            has_audio = True
                            print(f"  [编码] 音频转换成功")
                        else:
                            error_msg = stderr.decode('utf-8', errors='ignore')
                            print(f"  [编码] 音频转换失败: {error_msg}")
                            audio_file.close()
                            os.remove(audio_file.name)
                            audio_file = None
                    else:
                        print(f"  [编码] 音频为空，跳过")
                            
            except Exception as e:
                print(f"  [编码] 音频处理失败，将保存无音频视频: {e}")
                if audio_file:
                    try:
                        audio_file.close()
                        os.remove(audio_file.name)
                    except:
                        pass
                audio_file = None
                has_audio = False
        else:
            print(f"  [编码] 无音频输入")
        
        try:
            # 构建 FFmpeg 命令
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',
            ]
            
            # 如果有音频文件，添加音频输入
            if has_audio and audio_file:
                ffmpeg_cmd.extend(['-i', audio_file.name])
            
            # 添加输出参数
            ffmpeg_cmd.extend([
                '-c:v', codec_name,
                *codec_params,
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),              # 明确指定输出帧率
                '-vsync', 'cfr',             # 强制恒定帧率（CFR），避免可变帧率问题
            ])
            
            if has_audio:
                # 统一音频参数，确保所有视频的音频格式完全一致（关键！）
                ffmpeg_cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',  # 统一采样率为 44100Hz
                    '-ac', '2',      # 统一为立体声
                    '-shortest',     # 以较短的流为准
                ])
            
            ffmpeg_cmd.append(output_path)
            
            print(f"  [编码] FFmpeg 命令: {' '.join(ffmpeg_cmd[:10])}...")
            
            # 准备视频数据
            video_data = frames_np.tobytes()
            print(f"  [编码] 视频数据大小: {len(video_data) / 1024 / 1024:.2f} MB")
            print(f"  [编码] 开始编码...")
            
            # 执行 FFmpeg，使用 communicate 直接传入数据避免管道阻塞
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 使用 communicate 一次性传入所有数据，避免管道阻塞
            try:
                stdout, stderr = process.communicate(input=video_data, timeout=300)  # 5分钟超时
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise Exception("FFmpeg 编码超时（超过5分钟）")
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                print(f"  [编码] ✗ FFmpeg 错误:")
                print(f"  {error_msg}")
                raise Exception(f"FFmpeg 编码失败: {error_msg[:500]}")
            
            # 检查输出文件
            if not os.path.exists(output_path):
                raise Exception(f"输出文件未创建: {output_path}")
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise Exception(f"输出文件为空: {output_path}")
            
            print(f"  [编码] ✓ 编码完成: {output_path}")
            print(f"  [编码] 文件大小: {file_size / 1024 / 1024:.2f} MB")
                
        finally:
            # 清理临时音频文件
            if audio_file:
                try:
                    audio_file.close()
                    os.remove(audio_file.name)
                except:
                    pass
    
    def _check_nvidia_gpu(self):
        """检测是否支持 NVIDIA GPU 编码"""
        try:
            # 检查 NVIDIA 编码器是否可用
            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            output = result.stdout
            
            # 检查是否有 h264_nvenc 编码器
            has_nvenc = 'h264_nvenc' in output
            
            if has_nvenc:
                print(f"  [GPU] ✓ 检测到 NVIDIA GPU 编码器 (h264_nvenc)")
            else:
                print(f"  [GPU] ⚠ 未检测到 NVIDIA GPU 编码器，将使用 CPU")
            
            return has_nvenc
        except Exception as e:
            print(f"  [GPU] ⚠ GPU 检测失败: {e}，将使用 CPU")
            return False
    
    def _check_videos_format_consistency(self, video_list):
        """检查所有视频的格式是否一致（编码、分辨率、帧率、音频）"""
        print(f"  [检测] 检查 {len(video_list)} 个视频的格式一致性...")
        
        try:
            video_info_list = []
            
            # 获取所有视频的信息
            for video_path in video_list:
                probe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,width,height,r_frame_rate,pix_fmt',
                    '-show_entries', 'stream=codec_name:stream_tags=',
                    '-of', 'json',
                    video_path
                ]
                
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    print(f"  [检测] ⚠ 无法获取视频信息: {os.path.basename(video_path)}")
                    return False
                
                video_info = json.loads(result.stdout)
                if not video_info.get('streams'):
                    print(f"  [检测] ⚠ 无法解析视频流: {os.path.basename(video_path)}")
                    return False
                
                stream = video_info['streams'][0]
                video_info_list.append({
                    'path': video_path,
                    'codec': stream.get('codec_name', ''),
                    'width': stream.get('width', 0),
                    'height': stream.get('height', 0),
                    'fps': stream.get('r_frame_rate', ''),
                    'pix_fmt': stream.get('pix_fmt', '')
                })
            
            # 检查是否所有视频格式一致
            if not video_info_list:
                return False
            
            first_video = video_info_list[0]
            for i, video_info in enumerate(video_info_list[1:], 1):
                if (video_info['codec'] != first_video['codec'] or
                    video_info['width'] != first_video['width'] or
                    video_info['height'] != first_video['height'] or
                    video_info['fps'] != first_video['fps'] or
                    video_info['pix_fmt'] != first_video['pix_fmt']):
                    
                    print(f"  [检测] ✗ 视频格式不一致:")
                    print(f"    基准视频: {os.path.basename(first_video['path'])}")
                    print(f"      编码: {first_video['codec']}, 分辨率: {first_video['width']}x{first_video['height']}, 帧率: {first_video['fps']}, 像素格式: {first_video['pix_fmt']}")
                    print(f"    不同视频: {os.path.basename(video_info['path'])}")
                    print(f"      编码: {video_info['codec']}, 分辨率: {video_info['width']}x{video_info['height']}, 帧率: {video_info['fps']}, 像素格式: {video_info['pix_fmt']}")
                    return False
            
            print(f"  [检测] ✓ 所有视频格式一致:")
            print(f"    编码: {first_video['codec']}")
            print(f"    分辨率: {first_video['width']}x{first_video['height']}")
            print(f"    帧率: {first_video['fps']}")
            print(f"    像素格式: {first_video['pix_fmt']}")
            return True
            
        except Exception as e:
            print(f"  [检测] ⚠ 格式检测失败: {str(e)}")
            return False
    
    def _merge_all_videos(self, video_list, output_path):
        """一次性合并所有视频片段（使用FFmpeg concat demuxer + 流复制）"""
        print(f"  [合并] 开始合并 {len(video_list)} 个视频...")
        
        # 创建临时文件列表
        filelist_file = None
        try:
            # 使用临时文件存储文件列表
            import tempfile
            filelist_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            
            # 写入文件列表
            for video_path in video_list:
                # 使用绝对路径并转换为正斜杠（跨平台兼容）
                abs_path = os.path.abspath(video_path).replace('\\', '/')
                filelist_file.write(f"file '{abs_path}'\n")
            
            filelist_file.close()
            filelist_path = filelist_file.name
            
            print(f"  [合并] 临时文件列表: {filelist_path}")
            print(f"  [合并] 视频列表:")
            for i, video in enumerate(video_list, 1):
                print(f"    {i}. {os.path.basename(video)}")
            
            # 使用 FFmpeg concat demuxer 合并（参考 ComfyUI-FFmpeg）
            # 使用流复制模式（-c copy）速度快
            command = [
                'ffmpeg',
                '-y',                           # 覆盖输出文件
                '-f', 'concat',                 # 使用 concat demuxer
                '-safe', '0',                   # 允许不安全的文件路径
                '-i', filelist_path,            # 输入文件列表
                '-c', 'copy',                   # 流复制（不重新编码）
                '-fflags', '+genpts',           # 重新生成 PTS（修复时间戳）
                '-avoid_negative_ts', 'make_zero',  # 避免负时间戳
                output_path                     # 输出文件
            ]
            
            print(f"  [合并] 使用流复制模式（快速合并）...")
            print(f"  [合并] 执行命令: {' '.join(command[:8])}...")
            
            # 执行合并
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=600)
            
            # 检查返回码
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"  [合并] ✗ FFmpeg 错误:")
                print(f"  {error_msg[:500]}")
                raise ValueError(f"FFmpeg 合并失败: {error_msg[:500]}")
            
            # 检查输出文件
            if not os.path.exists(output_path):
                raise ValueError(f"输出文件未创建: {output_path}")
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise ValueError(f"输出文件为空: {output_path}")
            
            print(f"  [合并] ✓ 合并成功: {output_path}")
            print(f"  [合并] 最终文件大小: {file_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"  [合并] ✗ 合并失败: {str(e)}")
            raise
        
        finally:
            # 清理临时文件列表
            if filelist_file and os.path.exists(filelist_file.name):
                try:
                    os.remove(filelist_file.name)
                    print(f"  [合并] 已清理临时文件列表")
                except:
                    pass
    
    def _append_video(self, existing_video, new_video, audio=None):
        """将新视频追加到已存在的视频末尾"""
        print(f"  [合并] 开始合并视频...")
        print(f"  [合并] 已存在视频: {existing_video}")
        print(f"  [合并] 新视频: {new_video}")
        
        # 创建临时输出文件
        temp_output = existing_video + '.tmp.mp4'
        
        try:
            # 创建临时文件列表
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                concat_file = f.name
                # 使用绝对路径并转换为正斜杠（Windows 兼容性）
                existing_abs = os.path.abspath(existing_video).replace('\\', '/')
                new_abs = os.path.abspath(new_video).replace('\\', '/')
                f.write(f"file '{existing_abs}'\n")
                f.write(f"file '{new_abs}'\n")
            
            print(f"  [合并] Concat 文件: {concat_file}")
            
            try:
                # 完全重新编码模式（最可靠，避免累积错误）
                # 这样可以确保每次合并后的视频都是标准格式
                concat_cmd = [
                    'ffmpeg',
                    '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c:v', 'libx264',      # 视频重新编码
                    '-crf', '23',
                    '-preset', 'medium',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',          # 音频重新编码
                    '-b:a', '192k',
                    '-ar', '44100',         # 统一采样率
                    '-ac', '2',             # 统一为立体声
                    '-avoid_negative_ts', 'make_zero',  # 修正时间戳
                    '-fflags', '+genpts',   # 重新生成时间戳
                    temp_output
                ]
                
                print(f"  [合并] 使用完全重新编码模式（避免累积错误）...")
                result = subprocess.run(concat_cmd, capture_output=True, timeout=300)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  [合并] FFmpeg 错误: {error_msg}")
                    raise Exception(f"合并失败: {error_msg[:500]}")
                
                # 检查输出文件是否存在且有效
                if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
                    raise Exception("合并后的文件无效或为空")
                
                # 替换原文件
                os.replace(temp_output, existing_video)
                print(f"  [合并] 合并成功: {existing_video}")
                
                # 显示合并后的文件大小
                final_size = os.path.getsize(existing_video) / 1024 / 1024
                print(f"  [合并] 合并后文件大小: {final_size:.2f} MB")
                
            finally:
                # 清理临时文件列表
                if os.path.exists(concat_file):
                    os.remove(concat_file)
        
        except Exception as e:
            print(f"  [合并] 合并失败: {str(e)}")
            raise
        
        finally:
            # 清理临时输出文件
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except:
                    pass
    
    def _get_unique_filename(self, directory, filename):
        """生成唯一的文件名（如果文件已存在则添加序号）"""
        output_path = directory / filename
        
        if not output_path.exists():
            return str(output_path)
        
        # 文件已存在，生成带序号的新文件名
        base_name = output_path.stem
        extension = output_path.suffix
        counter = 1
        
        while True:
            new_filename = f"{base_name}_{counter:03d}{extension}"
            output_path = directory / new_filename
            if not output_path.exists():
                print(f"[视频合并器] 检测到同名文件，自动重命名为: {output_path.name}")
                return str(output_path)
            counter += 1
            if counter > 999:
                import time
                timestamp = int(time.time())
                new_filename = f"{base_name}_{timestamp}{extension}"
                output_path = directory / new_filename
                print(f"[视频合并器] 已存在超过999个同名文件，使用时间戳: {output_path.name}")
                return str(output_path)
    
    def _merge_videos_with_normalization(self, video_list, output_path, codec, quality):
        """合并视频并统一格式（基于第一个视频的参数）"""
        print(f"  [合并] 开始合并并统一格式...")
        print(f"  [合并] 视频数量: {len(video_list)}")
        
        try:
            # 获取第一个视频的参数作为基准
            first_video = video_list[0]
            print(f"  [合并] 基准视频: {os.path.basename(first_video)}")
            
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,codec_name',
                '-of', 'json',
                first_video
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
            
            if not video_info.get('streams'):
                raise Exception(f"无法获取视频信息: {first_video}")
            
            stream = video_info['streams'][0]
            target_width = stream.get('width', 1920)
            target_height = stream.get('height', 1080)
            fps_str = stream.get('r_frame_rate', '30/1')
            fps_parts = fps_str.split('/')
            target_fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            
            print(f"  [合并] 目标参数: {target_width}x{target_height} @ {target_fps:.2f} fps")
            
            # 检测 GPU 支持
            use_gpu = self._check_nvidia_gpu()
            
            # 创建临时文件列表
            filelist_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            
            try:
                # 写入文件列表
                for video_path in video_list:
                    abs_path = os.path.abspath(video_path).replace('\\', '/')
                    filelist_file.write(f"file '{abs_path}'\n")
                
                filelist_file.close()
                filelist_path = filelist_file.name
                
                print(f"  [合并] 临时文件列表: {filelist_path}")
                
                # 设置编码器参数
                if codec == "h264":
                    if use_gpu:
                        print(f"  [合并] 使用 NVIDIA GPU 加速 (h264_nvenc)")
                        codec_name = "h264_nvenc"
                        codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
                    else:
                        codec_name = "libx264"
                        codec_params = ["-crf", str(quality), "-preset", "medium"]
                elif codec == "h265":
                    if use_gpu:
                        print(f"  [合并] 使用 NVIDIA GPU 加速 (hevc_nvenc)")
                        codec_name = "hevc_nvenc"
                        codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
                    else:
                        codec_name = "libx265"
                        codec_params = ["-crf", str(quality), "-preset", "medium"]
                else:
                    codec_name = "libx264"
                    codec_params = ["-crf", str(quality), "-preset", "medium"]
                
                # 合并并统一格式（居中裁剪，无黑边）
                # 使用 scale + crop 实现居中裁剪：
                # 1. scale: 放大到目标尺寸（保持宽高比，确保覆盖整个画面）
                # 2. crop: 居中裁剪到目标尺寸
                command = [
                    'ffmpeg',
                    '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', filelist_path,
                    '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height},fps={target_fps}',
                    '-c:v', codec_name,
                    *codec_params,
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2',
                    output_path
                ]
                
                print(f"  [合并] 统一格式（居中裁剪）: {target_width}x{target_height} @ {target_fps:.2f} fps")
                print(f"  [合并] 执行合并...")
                
                result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=600)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  [合并] ✗ FFmpeg 错误: {error_msg[:500]}")
                    raise ValueError(f"FFmpeg 合并失败: {error_msg[:500]}")
                
                # 检查输出文件
                if not os.path.exists(output_path):
                    raise ValueError(f"输出文件未创建: {output_path}")
                
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    raise ValueError(f"输出文件为空: {output_path}")
                
                print(f"  [合并] ✓ 合并成功: {output_path}")
                print(f"  [合并] 最终文件大小: {file_size / 1024 / 1024:.2f} MB")
                
            finally:
                # 清理临时文件列表
                if os.path.exists(filelist_path):
                    try:
                        os.remove(filelist_path)
                    except:
                        pass
                        
        except Exception as e:
            print(f"  [合并] ✗ 合并失败: {str(e)}")
            raise
    
    def _merge_videos_with_normalization_and_audio(self, video_list, output_path, codec, quality, audio_file):
        """合并视频并统一格式，同时合并或替换音频（基于第一个视频的参数）"""
        print(f"  [合并] 开始合并并统一格式（包含音频）...")
        print(f"  [合并] 视频数量: {len(video_list)}")
        print(f"  [合并] 音频文件: {os.path.basename(audio_file)}")
        
        try:
            # 获取第一个视频的参数作为基准
            first_video = video_list[0]
            print(f"  [合并] 基准视频: {os.path.basename(first_video)}")
            
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,codec_name',
                '-of', 'json',
                first_video
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
            
            if not video_info.get('streams'):
                raise Exception(f"无法获取视频信息: {first_video}")
            
            stream = video_info['streams'][0]
            target_width = stream.get('width', 1920)
            target_height = stream.get('height', 1080)
            fps_str = stream.get('r_frame_rate', '30/1')
            fps_parts = fps_str.split('/')
            target_fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            
            print(f"  [合并] 目标参数: {target_width}x{target_height} @ {target_fps:.2f} fps")
            
            # 检测 GPU 支持
            use_gpu = self._check_nvidia_gpu()
            
            # 创建临时文件列表
            filelist_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            
            try:
                # 写入文件列表
                for video_path in video_list:
                    abs_path = os.path.abspath(video_path).replace('\\', '/')
                    filelist_file.write(f"file '{abs_path}'\n")
                
                filelist_file.close()
                filelist_path = filelist_file.name
                
                print(f"  [合并] 临时文件列表: {filelist_path}")
                
                # 设置编码器参数
                if codec == "h264":
                    if use_gpu:
                        print(f"  [合并] 使用 NVIDIA GPU 加速 (h264_nvenc)")
                        codec_name = "h264_nvenc"
                        codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
                    else:
                        codec_name = "libx264"
                        codec_params = ["-crf", str(quality), "-preset", "medium"]
                elif codec == "h265":
                    if use_gpu:
                        print(f"  [合并] 使用 NVIDIA GPU 加速 (hevc_nvenc)")
                        codec_name = "hevc_nvenc"
                        codec_params = ["-preset", "p4", "-cq", str(quality), "-b:v", "0"]
                    else:
                        codec_name = "libx265"
                        codec_params = ["-crf", str(quality), "-preset", "medium"]
                else:
                    codec_name = "libx264"
                    codec_params = ["-crf", str(quality), "-preset", "medium"]
                
                # 合并并统一格式（居中裁剪，无黑边），同时添加音频
                # 使用 scale + crop 实现居中裁剪：
                # 1. scale: 放大到目标尺寸（保持宽高比，确保覆盖整个画面）
                # 2. crop: 居中裁剪到目标尺寸
                # 音频处理：使用 -shortest 确保音频和视频同步，如果音频更长则截断，如果更短则视频静音
                command = [
                    'ffmpeg',
                    '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', filelist_path,
                    '-i', audio_file,
                    '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height},fps={target_fps}',
                    '-c:v', codec_name,
                    *codec_params,
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2',
                    '-map', '0:v:0',  # 使用第一个输入（视频）的视频流
                    '-map', '1:a:0',  # 使用第二个输入（音频文件）的音频流
                    '-shortest',      # 以较短的流为准
                    output_path
                ]
                
                print(f"  [合并] 统一格式（居中裁剪）: {target_width}x{target_height} @ {target_fps:.2f} fps")
                print(f"  [合并] 音频处理: 替换为外部音频文件")
                print(f"  [合并] 执行合并...")
                
                result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=600)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  [合并] ✗ FFmpeg 错误: {error_msg[:500]}")
                    raise ValueError(f"FFmpeg 合并失败: {error_msg[:500]}")
                
                # 检查输出文件
                if not os.path.exists(output_path):
                    raise ValueError(f"输出文件未创建: {output_path}")
                
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    raise ValueError(f"输出文件为空: {output_path}")
                
                print(f"  [合并] ✓ 合并成功（已添加音频）: {output_path}")
                print(f"  [合并] 最终文件大小: {file_size / 1024 / 1024:.2f} MB")
                
            finally:
                # 清理临时文件列表
                if os.path.exists(filelist_path):
                    try:
                        os.remove(filelist_path)
                    except:
                        pass
                        
        except Exception as e:
            print(f"  [合并] ✗ 合并失败: {str(e)}")
            raise

# --------------------------------------------------------------------------
# 视频格式转换和分割节点
# --------------------------------------------------------------------------
class VideoConvertAndSplit:
    """视频格式转换（如需要）并按时间分割"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "视频路径": ("STRING", {"default": "input/video.mp4", "multiline": False, "placeholder": "支持中文路径", "tooltip": "需要处理的视频文件路径。支持常见视频格式如MP4、AVI、MOV等。路径支持相对路径（基于ComfyUI根目录）和绝对路径。支持中文路径和文件名。视频编码建议使用H.264以获得最佳兼容性。"}),
                "输出目录": ("STRING", {"default": "output/长视频转换分割", "multiline": False, "placeholder": "支持中文路径", "tooltip": "分割后视频文件的输出目录。程序会自动创建目录（如果不存在）。支持相对路径和绝对路径，支持中文目录名。建议使用单独的输出目录，避免文件混乱。目录路径将作为输出信息返回，可连接到其他节点使用。"}),
                "片段时间": ("INT", {"default": 10, "min": 1, "max": 3600, "step": 1, "description": "每段视频的时长（秒）", "tooltip": "每个视频片段的目标时长（秒）。范围：1-3600秒（1小时）。较短的片段（如10-30秒）适合批量处理和快速预览，较长的片段（如60-300秒）适合高质量输出。程序会基于关键帧进行精确分割，保证片段完整性。分割时会自动插入关键帧以确保分割点准确。"}),
                "按短边缩放": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16, "description": "按短边缩放，0表示自动规范化为16的倍数（推荐）", "tooltip": "视频缩放设置。设为0时自动将视频尺寸规范化为16的倍数（推荐设置，确保兼容性）。设为其他值时，按短边缩放到指定像素值（如512、720、1080等）。建议使用标准分辨率如512、720、1080等。保持宽高比自动调整长边。步长为16以确保编码效率。"}),
                "强制帧率": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1, "description": "强制转换到目标帧率，0表示保持原帧率", "tooltip": "目标帧率设置。设为0时保持原视频帧率不变。设为其他值（如24、30、60）时强制转换视频到指定帧率。适合将高帧率视频转换为低帧率以减小文件大小，或将低帧率视频提升到标准帧率。注意：强制帧率转换需要重新编码，会增加处理时间和文件大小。"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("输出目录", "分割数量", "原视频帧率", "原文件名", "原音频路径")
    FUNCTION = "convert_and_split"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "自动转换视频格式为MP4（如需要）并按指定时间分割视频（基于关键帧分割）。原文件名输出可直接连接到视频自动合并器的最终文件名。自动检测输出目录，如果已有分割文件则跳过处理。"
    
    @classmethod
    def _get_cache_file(cls):
        """获取缓存文件路径"""
        cache_dir = Path(__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "video_convert_split_cache.json"
    
    @classmethod
    def _load_cache(cls):
        """加载缓存"""
        cache_file = cls._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    @classmethod
    def _save_cache(cls, cache_data):
        """保存缓存"""
        try:
            cache_file = cls._get_cache_file()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[视频转换分割] 保存缓存失败: {e}")
    
    @classmethod
    def _get_cache_key(cls, 视频路径, 输出目录, 片段时间, 按短边缩放, 转换为MP4, 强制帧率):
        """生成缓存键"""
        key_str = f"{视频路径}#{输出目录}#{片段时间}#{按短边缩放}#{转换为MP4}#{强制帧率}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def convert_and_split(self, 视频路径, 输出目录, 片段时间, 按短边缩放, 强制帧率):
        # 后台默认开启的选项
        转换为MP4 = True  # 默认开启转换为MP4（仅对非MP4格式生效）
        VFR转CFR = True  # 默认开启VFR转CFR
        插入关键帧 = True  # 默认开启插入关键帧
        
        try:
            print(f"\n{'='*60}")
            print(f"[视频转换分割] 开始处理...")
            print(f"[视频转换分割] 输入视频: {视频路径}")
            print(f"[视频转换分割] 输出目录: {输出目录}")
            print(f"[视频转换分割] 片段时间: {片段时间} 秒")
            print(f"[视频转换分割] 按短边缩放: {按短边缩放 if 按短边缩放 > 0 else '不缩放'}")
            print(f"[视频转换分割] 强制帧率: {强制帧率 if 强制帧率 > 0 else '保持原帧率'}")
            
            # 解析路径（相对路径基于ComfyUI根目录）
            视频路径 = resolve_path(视频路径)
            输出目录 = resolve_path(输出目录)
            
            print(f"[视频转换分割] 解析后视频路径: {视频路径}")
            print(f"[视频转换分割] 解析后输出目录: {输出目录}")
            
            # 检查输出目录是否已有分割文件
            output_dir = Path(输出目录)
            if output_dir.exists():
                # 检查是否有分割文件（0001.mp4, 0002.mp4等）或音频文件（0000_audio.*）
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
                existing_files = []
                for ext in video_extensions:
                    existing_files.extend(output_dir.glob(f"[0-9][0-9][0-9][0-9]{ext}"))
                # 也检查音频文件
                existing_files.extend(output_dir.glob("0000_audio.*"))
                
                if existing_files:
                    print(f"[视频转换分割] ⏭️ 检测到输出目录已有 {len(existing_files)} 个文件，跳过处理")
                    print(f"[视频转换分割] 提示：如需重新处理，请手动删除输出目录中的文件")
                    print(f"[视频转换分割] 输出目录: {输出目录}")
                    
                    # 尝试读取已有的信息
                    video_path = Path(视频路径)
                    original_filename = video_path.stem
                    
                    # 统计分割文件数量
                    split_files = []
                    for ext in video_extensions:
                        split_files.extend(output_dir.glob(f"[0-9][0-9][0-9][0-9]{ext}"))
                    split_count = len(split_files)
                    
                    # 查找音频文件
                    audio_files = list(output_dir.glob("0000_audio.*"))
                    audio_path = str(audio_files[0]) if audio_files else ""
                    
                    print(f"[视频转换分割] 已有分割数量: {split_count}")
                    print(f"[视频转换分割] 原文件名: {original_filename}")
                    if audio_path:
                        print(f"[视频转换分割] 音频文件: {Path(audio_path).name}")
                    print(f"{'='*60}\n")
                    
                    return (输出目录, split_count, 0.0, original_filename, audio_path)
            
            # 检查视频文件是否存在
            video_path = Path(视频路径)
            if not video_path.exists() or not video_path.is_file():
                print(f"[视频转换分割] ✗ 错误: 视频文件不存在: {视频路径}")
                return (输出目录, 0, 0.0, "", "")
            
            # 创建输出目录
            output_dir = Path(输出目录)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取视频信息（包括帧率、帧数、尺寸、时长）
            print(f"[视频转换分割] 获取视频信息...")
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,r_frame_rate,nb_frames,width,height:format=duration',
                '-of', 'json',
                str(video_path)
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            
            # 获取视频信息
            stream = probe_data.get('streams', [{}])[0]
            duration = float(probe_data.get('format', {}).get('duration', 0))
            video_codec = stream.get('codec_name', 'unknown')
            
            # 解析帧率
            fps_str = stream.get('r_frame_rate', '30/1')
            fps_parts = fps_str.split('/')
            original_fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            
            # 获取帧数
            nb_frames = int(stream.get('nb_frames', 0))
            if nb_frames == 0 and duration > 0:
                nb_frames = int(duration * original_fps)
            
            # 获取视频尺寸
            orig_width = int(stream.get('width', 0))
            orig_height = int(stream.get('height', 0))
            
            # 显示视频信息
            print(f"[视频转换分割] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"[视频转换分割] 📹 视频信息:")
            print(f"[视频转换分割]   • 时长: {duration:.2f} 秒")
            print(f"[视频转换分割]   • 帧率: {original_fps:.2f} fps")
            print(f"[视频转换分割]   • 帧数: {nb_frames} 帧")
            print(f"[视频转换分割]   • 尺寸: {orig_width}x{orig_height}")
            print(f"[视频转换分割]   • 编码: {video_codec}")
            print(f"[视频转换分割] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # 获取文件名和扩展名
            file_full_name = video_path.name
            file_name = video_path.stem
            file_ext = video_path.suffix
            
            # 保存原文件名（不含扩展名）用于输出
            original_filename = file_name
            
            # 检查是否为 MP4 格式
            is_mp4 = file_ext.lower() == '.mp4'
            
            print(f"[视频转换分割] 文件格式: {file_ext}")
            
            # 解析强制帧率
            target_fps = None
            if 强制帧率 > 0:
                target_fps = float(强制帧率)
            
            # 决定处理流程（优化版：减少编码次数）
            # 检查尺寸是否需要规范化（不是16的倍数）
            need_normalize = (orig_width % 16 != 0) or (orig_height % 16 != 0)
            need_scale = 按短边缩放 > 0 or need_normalize  # 缩放或规范化
            need_convert = 转换为MP4 and not is_mp4
            need_reduce_fps = target_fps is not None
            need_cfr = VFR转CFR
            is_h264 = video_codec.lower() in ['h264', 'avc']
            
            temp_files = []  # 记录临时文件
            source_video = str(video_path)
            output_ext = '.mp4'
            step_num = 0
            
            # 步骤0: 缩放或规范化尺寸（如果需要）
            if need_scale:
                step_num += 1
                if 按短边缩放 > 0:
                    print(f"[视频转换分割] 步骤{step_num}: 按短边缩放到 {按短边缩放}...")
                else:
                    print(f"[视频转换分割] 步骤{step_num}: 规范化尺寸为16的倍数...")
                
                scaled_video = output_dir / f"{file_name}_scaled{file_ext}"
                self._scale_video(str(video_path), str(scaled_video), 按短边缩放, orig_width, orig_height)
                source_video = str(scaled_video)
                temp_files.append(source_video)
                
                if 按短边缩放 > 0:
                    print(f"[视频转换分割] ✓ 缩放完成")
                else:
                    print(f"[视频转换分割] ✓ 尺寸规范化完成")
            
            # 优化流程：合并CFR转换和格式转换，避免二次编码
            # 情况1: 非MP4或非H264 -> 需要格式转换，同时进行CFR转换和关键帧插入
            if need_convert or not is_h264:
                step_num += 1
                if need_convert and not is_h264:
                    print(f"[视频转换分割] 步骤{step_num}: 转换为 MP4/H264 + CFR转换 + 插入关键帧（一次编码完成）...")
                elif need_convert:
                    print(f"[视频转换分割] 步骤{step_num}: 转换为 MP4 + CFR转换 + 插入关键帧（一次编码完成）...")
                else:
                    print(f"[视频转换分割] 步骤{step_num}: 转换为 H264 + CFR转换 + 插入关键帧（一次编码完成）...")
                
                converted_video = output_dir / f"{file_name}_processed.mp4"
                
                # 合并处理：格式转换 + CFR转换 + 关键帧插入（一次编码）
                self._convert_and_prepare_for_split(
                    source_video, 
                    str(converted_video), 
                    original_fps,
                    segment_time=片段时间 if 插入关键帧 else None,
                    target_fps=target_fps
                )
                
                # 清理之前的临时文件
                if source_video in temp_files:
                    try:
                        os.remove(source_video)
                        temp_files.remove(source_video)
                    except:
                        pass
                
                source_video = str(converted_video)
                temp_files.append(source_video)
                print(f"[视频转换分割] ✓ 处理完成（已优化为一次编码）")
            
            # 情况2: 已是MP4/H264 -> 只需CFR转换和关键帧插入
            elif need_cfr or 插入关键帧:
                step_num += 1
                if 插入关键帧:
                    print(f"[视频转换分割] 步骤{step_num}: CFR转换 + 插入关键帧（每 {片段时间} 秒）...")
                else:
                    print(f"[视频转换分割] 步骤{step_num}: CFR转换（优先流复制）...")
                
                cfr_video = output_dir / f"{file_name}_cfr.mp4"
                self._convert_vfr_to_cfr(
                    source_video, 
                    str(cfr_video), 
                    original_fps, 
                    segment_time=片段时间, 
                    force_keyframes=插入关键帧
                )
                
                # 清理之前的临时文件
                if source_video in temp_files:
                    try:
                        os.remove(source_video)
                        temp_files.remove(source_video)
                    except:
                        pass
                
                source_video = str(cfr_video)
                temp_files.append(source_video)
                print(f"[视频转换分割] ✓ CFR转换完成")
            else:
                print(f"[视频转换分割] 视频已是 MP4/H264 格式，无需转换")
            
            # 最后步骤: 分割视频（基于关键帧）
            step_num += 1
            print(f"[视频转换分割] 步骤{step_num}: 分割视频（基于关键帧，每段约 {片段时间} 秒）...")
            split_count = self._split_video_by_segment(source_video, str(output_dir), file_name, 片段时间, output_ext)
            
            # 清理所有临时文件
            if temp_files:
                print(f"[视频转换分割] 清理临时文件...")
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print(f"  - 已删除: {os.path.basename(temp_file)}")
                    except Exception as e:
                        print(f"  - 删除失败: {os.path.basename(temp_file)} - {e}")
            
            print(f"[视频转换分割] ✓ 完成! 共分割为 {split_count} 个片段")
            print(f"[视频转换分割] 原文件名输出: {original_filename}")
            
            # 提取原音频（用于视频自动合并器）
            audio_output_path = ""
            try:
                print(f"[视频转换分割] 提取原音频...")
                
                # 确定音频来源（使用原始视频，保留原始音频质量）
                audio_source = str(video_path)
                
                # 提取音频
                audio_output_path = self._extract_audio(audio_source, str(output_dir), file_name)
                
                if audio_output_path:
                    print(f"[视频转换分割] ✓ 音频提取完成: {os.path.basename(audio_output_path)}")
                else:
                    print(f"[视频转换分割] ⚠️ 未提取音频（视频可能无音频流）")
            except Exception as e:
                print(f"[视频转换分割] ⚠️ 音频提取失败: {str(e)}")
                audio_output_path = ""
            
            print(f"{'='*60}\n")
            
            return (输出目录, split_count, original_fps, original_filename, audio_output_path)
            
        except Exception as e:
            print(f"[视频转换分割] ✗ 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return (输出目录, 0, 0.0, "", "")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """控制节点缓存行为 - 检查输出目录是否已有文件"""
        输出目录 = kwargs.get('输出目录', '')
        
        # 解析路径
        输出目录 = resolve_path(输出目录)
        
        output_dir = Path(输出目录)
        if output_dir.exists():
            # 检查是否有分割文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            existing_files = []
            for ext in video_extensions:
                existing_files.extend(output_dir.glob(f"[0-9][0-9][0-9][0-9]{ext}"))
            
            if existing_files:
                # 有文件，返回固定值（不重新执行）
                return "has_files"
        
        # 没有文件，需要执行
        return float("NaN")
    
    def _split_video_by_segment(self, input_path, output_dir, file_name, segment_time, output_ext):
        """使用 FFmpeg segment muxer 按时间分割视频（基于关键帧，精确分割）
        
        注意：此函数假设输入视频已经在关键帧位置插入了关键帧（通过 CFR 转换）
        这样可以使用 -c copy 快速分割，同时保证分割点精确
        """
        try:
            # 构建输出文件名模板（4位数字序号）
            output_pattern = os.path.join(output_dir, f"%04d{output_ext}")
            
            # 使用 FFmpeg segment muxer 分割
            # 由于 CFR 转换时已插入关键帧，这里可以安全使用 -c copy
            command = [
                'ffmpeg',
                '-i', input_path,                    # 输入视频
                '-f', 'segment',                     # 使用 segment muxer
                '-segment_time', str(segment_time),  # 每段时长（秒）
                '-segment_start_number', '1',        # 从1开始编号（而不是0）
                '-reset_timestamps', '1',            # 重置每个片段的时间戳为0（避免合并时时间戳混乱）
                '-c', 'copy',                        # 流复制（不重新编码，快速且无损）
                output_pattern                       # 输出文件名模板
            ]
            
            print(f"  [分割] 执行命令: ffmpeg -i ... -f segment -segment_time {segment_time} -reset_timestamps 1 -c copy ...")
            
            # 执行命令
            result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            # 检查返回码
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"  [分割] ✗ FFmpeg 错误: {error_msg[:500]}")
                raise ValueError(f"FFmpeg 分割失败: {error_msg[:500]}")
            
            # 统计生成的文件数量
            output_dir_path = Path(output_dir)
            # 匹配4位数字的文件名：0001.mp4, 0002.mp4, ...
            split_files = sorted(output_dir_path.glob(f"[0-9][0-9][0-9][0-9]{output_ext}"))
            split_count = len(split_files)
            
            # 显示分割结果
            print(f"  [分割] ✓ 成功分割为 {split_count} 个片段（精确分割，基于关键帧）:")
            for i, file in enumerate(split_files, 1):
                file_size = file.stat().st_size / 1024 / 1024  # MB
                print(f"    片段 {i}: {file.name} ({file_size:.2f} MB)")
            
            return split_count
            
        except Exception as e:
            print(f"  [分割] ✗ 分割失败: {str(e)}")
            raise
    
    def _extract_audio(self, input_path, output_dir, file_name):
        """提取视频中的音频，如果是AAC/MP3则直接复制，否则转换为AAC
        
        Args:
            input_path: 输入视频路径
            output_dir: 输出目录
            file_name: 文件名（不含扩展名）
        
        Returns:
            str: 音频文件路径，如果失败或无音频则返回空字符串
        """
        try:
            print(f"  [音频] 检测音频流...")
            
            # 检测音频流信息
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'json',
                input_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  [音频] 未检测到音频流")
                return ""
            
            audio_info = json.loads(result.stdout)
            
            if not audio_info.get('streams'):
                print(f"  [音频] 视频无音频流")
                return ""
            
            audio_codec = audio_info['streams'][0].get('codec_name', 'unknown')
            print(f"  [音频] 音频编码: {audio_codec}")
            
            # 确定输出格式和编码方式（使用0000命名以区分视频文件）
            if audio_codec in ['aac', 'mp3']:
                # AAC 或 MP3，直接复制
                output_ext = '.aac' if audio_codec == 'aac' else '.mp3'
                output_path = os.path.join(output_dir, f"0000_audio{output_ext}")
                
                print(f"  [音频] 直接复制音频流（{audio_codec}）...")
                extract_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-vn',                  # 不处理视频
                    '-acodec', 'copy',      # 音频流复制
                    output_path
                ]
            else:
                # 其他格式，转换为 AAC
                output_path = os.path.join(output_dir, f"0000_audio.aac")
                
                print(f"  [音频] 转换音频为 AAC 格式...")
                extract_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-vn',                  # 不处理视频
                    '-acodec', 'aac',       # 转换为 AAC
                    '-b:a', '192k',         # 比特率 192kbps
                    '-ar', '44100',         # 采样率 44.1kHz
                    '-ac', '2',             # 立体声
                    output_path
                ]
            
            # 执行提取
            result = subprocess.run(extract_cmd, capture_output=True)
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"  [音频] ✗ 提取失败: {error_msg[:200]}")
                return ""
            
            # 检查输出文件
            if not os.path.exists(output_path):
                print(f"  [音频] ✗ 输出文件未创建")
                return ""
            
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"  [音频] ✓ 音频提取成功: {os.path.basename(output_path)} ({file_size:.2f} MB)")
            
            return output_path
            
        except Exception as e:
            print(f"  [音频] ✗ 提取失败: {str(e)}")
            return ""
    
    def _convert_vfr_to_cfr(self, input_path, output_path, fps, segment_time=10, force_keyframes=False):
        """将可变帧率(VFR)视频转换为恒定帧率(CFR)
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            fps: 目标帧率
            segment_time: 分段时长（秒），用于插入关键帧，默认10秒
            force_keyframes: 是否强制插入关键帧（需要重新编码）
        """
        try:
            if force_keyframes:
                print(f"  [CFR] 转换 VFR → CFR（每 {segment_time} 秒插入关键帧）...")
            else:
                print(f"  [CFR] 转换 VFR → CFR（优先流复制）...")
            print(f"  [CFR] 输入: {os.path.basename(input_path)}")
            print(f"  [CFR] 输出: {os.path.basename(output_path)}")
            print(f"  [CFR] 目标帧率: {fps} fps")
            
            # 检测 GPU 支持
            use_gpu = self._check_nvidia_gpu()
            
            # 策略1: 如果不需要插入关键帧，优先尝试流复制（最快，无损）
            # 如果分割结果不理想，会触发自动重试并插入关键帧
            if not force_keyframes:
                print(f"  [CFR] 尝试流复制模式（快速，无损）...")
                cfr_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-c:v', 'copy',              # 视频流复制（不重新编码，保持质量）
                    '-c:a', 'copy',              # 音频流复制
                    '-r', str(fps),              # 指定帧率
                    '-vsync', 'cfr',             # 强制恒定帧率
                    output_path
                ]
                
                result = subprocess.run(cfr_cmd, capture_output=True)
                
                if result.returncode == 0:
                    # 流复制成功
                    file_size = os.path.getsize(output_path) / 1024 / 1024
                    print(f"  [CFR] ✓ CFR 转换完成（流复制，无损）")
                    print(f"  [CFR] 输出文件大小: {file_size:.2f} MB")
                    print(f"  [CFR] 提示：如果分割结果不理想，将自动重试并插入关键帧")
                    return
                else:
                    # 流复制失败，继续尝试重新编码
                    print(f"  [CFR] 流复制失败，切换到重新编码模式...")
            
            # 策略2: 重新编码（插入关键帧或流复制失败时）
            if use_gpu:
                print(f"  [CFR] 使用 NVIDIA GPU 加速编码...")
                
                # 优化的 GPU 编码参数（减小体积）
                cfr_cmd = [
                    'ffmpeg',
                    '-y',
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda',
                    '-i', input_path,
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',             # 平衡预设
                    '-profile:v', 'high',        # 使用 High Profile（更好的压缩）
                    '-rc', 'vbr',                # 可变比特率模式
                    '-cq', '23',                 # 质量参数（23=高质量，体积适中）
                    '-b:v', '0',                 # VBR 模式
                    '-maxrate', '10M',           # 最大比特率限制
                    '-bufsize', '20M',           # 缓冲区大小
                ]
                
                # 如果需要插入关键帧
                if force_keyframes:
                    # 计算GOP大小 = 帧率 × 片段时间
                    gop_size = int(fps * segment_time)
                    print(f"  [CFR] 设置GOP: {gop_size}帧 (帧率{fps} × {segment_time}秒)")
                    cfr_cmd.extend([
                        '-forced-idr', '1',
                        '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})',
                        '-g', str(gop_size),  # 设置GOP大小
                    ])
                
                cfr_cmd.extend([
                    '-c:a', 'copy',
                    '-r', str(fps),
                    '-vsync', 'cfr',
                    output_path
                ])
                
                result = subprocess.run(cfr_cmd, capture_output=True)
                
                if result.returncode != 0:
                    # GPU 编码失败，降级到 CPU
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  [CFR] GPU 编码失败: {error_msg[:200]}")
                    print(f"  [CFR] 降级到 CPU 编码...")
                    use_gpu = False
            
            # 策略3: 使用 CPU 编码（降级方案）
            if not use_gpu:
                print(f"  [CFR] 使用 CPU 编码...")
                
                # 优化的 CPU 编码参数（减小体积）
                cfr_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-crf', '23',                # CRF 23（高质量，体积适中）
                    '-preset', 'medium',         # 平衡预设（比 fast 压缩率更好）
                    '-profile:v', 'high',        # High Profile
                    '-level', '4.1',             # 兼容性级别
                ]
                
                # 如果需要插入关键帧
                if force_keyframes:
                    # 计算GOP大小 = 帧率 × 片段时间
                    gop_size = int(fps * segment_time)
                    print(f"  [CFR] 设置GOP: {gop_size}帧 (帧率{fps} × {segment_time}秒)")
                    cfr_cmd.extend([
                        '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})',
                        '-g', str(gop_size),  # 设置GOP大小
                    ])
                
                cfr_cmd.extend([
                    '-c:a', 'copy',
                    '-r', str(fps),
                    '-vsync', 'cfr',
                    output_path
                ])
                
                result = subprocess.run(cfr_cmd, capture_output=True)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  [CFR] ✗ FFmpeg 错误: {error_msg[:500]}")
                    raise Exception(f"CFR 转换失败: {error_msg[:500]}")
            
            # 检查输出文件
            if not os.path.exists(output_path):
                raise Exception(f"输出文件未创建: {output_path}")
            
            file_size = os.path.getsize(output_path) / 1024 / 1024
            if force_keyframes:
                print(f"  [CFR] ✓ CFR 转换完成（已插入关键帧，每 {segment_time} 秒一个）")
            else:
                print(f"  [CFR] ✓ CFR 转换完成（重新编码）")
            print(f"  [CFR] 输出文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"  [CFR] ✗ CFR 转换失败: {str(e)}")
            raise
    
    def _force_fps(self, input_path, output_path, target_fps):
        """强制视频帧率（使用FFmpeg fps滤镜，参考VideoHelperSuite实现），优先使用GPU加速"""
        try:
            print(f"  [帧率] 使用FFmpeg fps滤镜强制帧率...")
            print(f"  [帧率] 输入: {os.path.basename(input_path)}")
            print(f"  [帧率] 输出: {os.path.basename(output_path)}")
            print(f"  [帧率] 强制到: {target_fps} fps")
            
            # 检测 GPU 支持
            use_gpu = self._check_nvidia_gpu()
            
            if use_gpu:
                # 使用 NVIDIA GPU 加速编码
                print(f"  [帧率] 使用 NVIDIA GPU 加速编码...")
                force_fps_cmd = [
                    'ffmpeg',
                    '-y',
                    '-hwaccel', 'cuda',                    # 启用 CUDA 硬件加速
                    '-hwaccel_output_format', 'cuda',
                    '-i', input_path,
                    '-vf', f'fps=fps={target_fps}',        # 使用 fps 滤镜强制帧率（参考VideoHelperSuite）
                    '-c:v', 'h264_nvenc',                  # NVIDIA GPU H.264 编码器
                    '-preset', 'p4',                       # GPU 预设：p1(快)-p7(慢)，p4 平衡
                    '-cq', '18',                           # 质量参数（类似 CRF）
                    '-b:v', '0',                           # 使用 CQ 模式
                    '-c:a', 'copy',                        # 音频流复制（不改变）
                    output_path
                ]
            else:
                # 使用 CPU 编码（降级方案）
                print(f"  [帧率] 使用 CPU 编码...")
                force_fps_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-vf', f'fps=fps={target_fps}',        # 使用 fps 滤镜强制帧率（参考VideoHelperSuite）
                    '-c:v', 'libx264',                     # 视频重新编码
                    '-crf', '18',                          # 高质量
                    '-preset', 'fast',
                    '-c:a', 'copy',                        # 音频流复制（不改变）
                    output_path
                ]
            
            print(f"  [帧率] 执行命令: {' '.join(force_fps_cmd[:8])}...")
            
            result = subprocess.run(force_fps_cmd, capture_output=True)
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"  [帧率] ✗ FFmpeg 错误:")
                print(f"  {error_msg[:500]}")
                
                # 如果GPU编码失败，尝试降级到CPU
                if use_gpu:
                    print(f"  [帧率] GPU 编码失败，尝试使用 CPU 编码...")
                    force_fps_cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', input_path,
                        '-vf', f'fps=fps={target_fps}',
                        '-c:v', 'libx264',
                        '-crf', '18',
                        '-preset', 'fast',
                        '-c:a', 'copy',
                        output_path
                    ]
                    result = subprocess.run(force_fps_cmd, capture_output=True)
                    if result.returncode != 0:
                        error_msg = result.stderr.decode('utf-8', errors='ignore')
                        raise Exception(f"强制帧率失败: {error_msg[:500]}")
                else:
                    raise Exception(f"强制帧率失败: {error_msg[:500]}")
            
            # 检查输出文件
            if not os.path.exists(output_path):
                raise Exception(f"输出文件未创建: {output_path}")
            
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"  [帧率] ✓ 强制帧率完成")
            print(f"  [帧率] 输出文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"  [帧率] ✗ 强制帧率失败: {str(e)}")
            raise
    
    def _scale_video(self, input_path, output_path, scale_short_side, orig_width, orig_height):
        """按短边缩放视频，保持宽高比
        
        Args:
            scale_short_side: 目标短边尺寸，0表示仅规范化为16的倍数（不改变尺寸）
        """
        try:
            print(f"  [缩放] 使用FFmpeg处理视频...")
            print(f"  [缩放] 输入: {os.path.basename(input_path)}")
            print(f"  [缩放] 输出: {os.path.basename(output_path)}")
            print(f"  [缩放] 原始尺寸: {orig_width}x{orig_height}")
            
            # 如果scale_short_side为0，表示仅规范化为16的倍数（不改变尺寸）
            if scale_short_side == 0:
                # 规范化模式：调整为16的倍数
                width = orig_width - (orig_width % 16)
                height = orig_height - (orig_height % 16)
                
                # 检查是否需要调整
                if width == orig_width and height == orig_height:
                    print(f"  [缩放] 尺寸已是16的倍数，无需调整")
                    # 直接复制文件（流复制，快速）
                    import shutil
                    shutil.copy2(input_path, output_path)
                    return
                else:
                    print(f"  [缩放] 规范化模式: 调整为16的倍数")
                    print(f"  [缩放] 目标尺寸: {width}x{height} (规范化)")
            else:
                # 缩放模式：按短边缩放
                short_side = min(orig_width, orig_height)
                long_side = max(orig_width, orig_height)
                scale_ratio = scale_short_side / short_side
                
                if orig_width < orig_height:
                    # 宽度是短边
                    width = scale_short_side
                    height = int(orig_height * scale_ratio)
                else:
                    # 高度是短边
                    height = scale_short_side
                    width = int(orig_width * scale_ratio)
                
                # 确保尺寸是16的倍数（视频编码最佳性能）
                width = width - (width % 16)
                height = height - (height % 16)
                
                print(f"  [缩放] 缩放模式: 短边 {scale_short_side}")
                print(f"  [缩放] 目标尺寸: {width}x{height}")
            
            print(f"  [缩放] 目标尺寸: {width}x{height} (短边: {scale_short_side})")
            
            # 检测 GPU 支持
            use_gpu = self._check_nvidia_gpu()
            
            if use_gpu:
                # 使用 NVIDIA GPU 加速编码
                print(f"  [缩放] 使用 NVIDIA GPU 加速编码...")
                scale_cmd = [
                    'ffmpeg',
                    '-y',
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda',
                    '-i', input_path,
                    '-vf', f'scale_cuda={width}:{height}',
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-cq', '23',
                    '-b:v', '0',
                    '-c:a', 'copy',
                    output_path
                ]
            else:
                # 使用 CPU 编码
                print(f"  [缩放] 使用 CPU 编码...")
                scale_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-vf', f'scale={width}:{height}',
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-c:a', 'copy',
                    output_path
                ]
            
            print(f"  [缩放] 执行命令: {' '.join(scale_cmd[:8])}...")
            
            result = subprocess.run(scale_cmd, capture_output=True)
            
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                print(f"  [缩放] ✗ FFmpeg 错误:")
                print(f"  {error_msg[:500]}")
                
                # 如果GPU编码失败，尝试降级到CPU
                if use_gpu:
                    print(f"  [缩放] GPU 编码失败，尝试使用 CPU 编码...")
                    scale_cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', input_path,
                        '-vf', f'scale={width}:{height}',
                        '-c:v', 'libx264',
                        '-crf', '23',
                        '-preset', 'medium',
                        '-c:a', 'copy',
                        output_path
                    ]
                    result = subprocess.run(scale_cmd, capture_output=True)
                    if result.returncode != 0:
                        error_msg = result.stderr.decode('utf-8', errors='ignore')
                        raise Exception(f"视频缩放失败: {error_msg[:500]}")
                else:
                    raise Exception(f"视频缩放失败: {error_msg[:500]}")
            
            # 检查输出文件
            if not os.path.exists(output_path):
                raise Exception(f"输出文件未创建: {output_path}")
            
            file_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"  [缩放] ✓ 缩放完成")
            print(f"  [缩放] 输出文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"  [缩放] ✗ 缩放失败: {str(e)}")
            raise
    
    def _check_nvidia_gpu(self):
        """检测是否支持 NVIDIA GPU 编码"""
        try:
            # 检查 NVIDIA 编码器是否可用
            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            output = result.stdout
            
            # 检查是否有 h264_nvenc 编码器
            has_nvenc = 'h264_nvenc' in output
            
            if has_nvenc:
                print(f"  [转换] ✓ 检测到 NVIDIA GPU 编码器 (h264_nvenc)")
            else:
                print(f"  [转换] ⚠ 未检测到 NVIDIA GPU 编码器，将使用 CPU")
            
            return has_nvenc
        except:
            return False
    
    def _convert_and_prepare_for_split(self, input_path, output_path, fps, segment_time=None, target_fps=None):
        """一次编码完成：格式转换 + CFR转换 + 关键帧插入 + 帧率调整
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径（MP4格式）
            fps: 原始帧率
            segment_time: 关键帧间隔（秒），如果提供则插入关键帧
            target_fps: 目标帧率，如果提供则调整帧率
        """
        # 检测 GPU 支持
        use_gpu = self._check_nvidia_gpu()
        
        # 确定最终帧率
        final_fps = target_fps if target_fps is not None else fps
        
        if use_gpu:
            # 使用 NVIDIA GPU 加速编码
            print(f"  [处理] 使用 NVIDIA GPU 加速编码...")
            convert_cmd = [
                'ffmpeg',
                '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'cuda',
                '-i', input_path,
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-profile:v', 'high',
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '0',
                '-maxrate', '10M',
                '-bufsize', '20M',
            ]
            
            # 插入关键帧（如果指定）
            if segment_time is not None:
                # 计算GOP大小 = 帧率 × 片段时间
                gop_size = int(final_fps * segment_time)
                print(f"  [处理] 插入关键帧: 每 {segment_time} 秒 (GOP={gop_size}帧)")
                convert_cmd.extend([
                    '-forced-idr', '1',
                    '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})',
                    '-g', str(gop_size),  # 设置GOP大小
                ])
            
            convert_cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '44100',
                '-ac', '2',
                '-r', str(final_fps),
                '-vsync', 'cfr',
                output_path
            ])
        else:
            # 使用 CPU 编码
            print(f"  [处理] 使用 CPU 编码...")
            convert_cmd = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-profile:v', 'high',
                '-level', '4.1',
            ]
            
            # 插入关键帧（如果指定）
            if segment_time is not None:
                # 计算GOP大小 = 帧率 × 片段时间
                gop_size = int(final_fps * segment_time)
                print(f"  [处理] 插入关键帧: 每 {segment_time} 秒 (GOP={gop_size}帧)")
                convert_cmd.extend([
                    '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})',
                    '-g', str(gop_size),  # 设置GOP大小
                ])
            
            convert_cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '44100',
                '-ac', '2',
                '-r', str(final_fps),
                '-vsync', 'cfr',
                output_path
            ])
        
        result = subprocess.run(convert_cmd, capture_output=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            
            # 如果 GPU 编码失败，降级到 CPU
            if use_gpu and 'nvenc' in error_msg.lower():
                print(f"  [处理] ⚠ GPU 编码失败，降级到 CPU 编码...")
                convert_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-profile:v', 'high',
                    '-level', '4.1',
                ]
                
                if segment_time is not None:
                    convert_cmd.extend([
                        '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})'
                    ])
                
                convert_cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2',
                    '-r', str(final_fps),
                    '-vsync', 'cfr',
                    output_path
                ])
                
                result = subprocess.run(convert_cmd, capture_output=True)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    raise Exception(f"视频处理失败: {error_msg[:500]}")
            else:
                raise Exception(f"视频处理失败: {error_msg[:500]}")
        
        # 检查输出文件
        if not os.path.exists(output_path):
            raise Exception(f"输出文件未创建: {output_path}")
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  [处理] 输出文件大小: {file_size:.2f} MB")
    
    def _convert_to_mp4(self, input_path, output_path, segment_time=None):
        """转换视频为 MP4 格式，优先使用 GPU 加速
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            segment_time: 关键帧间隔（秒），如果提供则插入关键帧
        """
        # 检测 GPU 支持
        use_gpu = self._check_nvidia_gpu()
        
        if use_gpu:
            # 使用 NVIDIA GPU 加速编码
            print(f"  [转换] 使用 NVIDIA GPU 加速编码...")
            convert_cmd = [
                'ffmpeg',
                '-y',
                '-hwaccel', 'cuda',             # 启用 CUDA 硬件加速
                '-hwaccel_output_format', 'cuda',
                '-i', input_path,
                '-c:v', 'h264_nvenc',           # NVIDIA GPU H.264 编码器
                '-preset', 'p4',                # GPU 预设：p1(快)-p7(慢)，p4 平衡
                '-cq', '23',                    # 质量参数（类似 CRF）
                '-b:v', '0',                    # 使用 CQ 模式
            ]
            
            # 如果指定了关键帧间隔，插入关键帧
            if segment_time is not None:
                # 需要先获取视频帧率来计算GOP
                # 这里使用一个合理的默认值（30fps）
                # 实际应用中，应该从视频信息中获取
                default_fps = 30.0
                gop_size = int(default_fps * segment_time)
                print(f"  [转换] 保持关键帧间隔: 每 {segment_time} 秒 (GOP≈{gop_size}帧)")
                convert_cmd.extend([
                    '-forced-idr', '1',
                    '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})',
                    '-g', str(gop_size),  # 设置GOP大小
                ])
            
            convert_cmd.extend([
                '-c:a', 'aac',                  # 音频编码为 AAC
                '-b:a', '192k',
                '-ar', '44100',                 # 统一采样率
                '-ac', '2',                     # 统一为立体声
                output_path
            ])
        else:
            # 使用 CPU 编码（降级方案）
            print(f"  [转换] 使用 CPU 编码...")
            convert_cmd = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-c:v', 'libx264',              # CPU H.264 编码器
                '-crf', '23',                   # 高质量
                '-preset', 'medium',
            ]
            
            # 如果指定了关键帧间隔，插入关键帧
            if segment_time is not None:
                # 需要先获取视频帧率来计算GOP
                # 这里使用一个合理的默认值（30fps）
                default_fps = 30.0
                gop_size = int(default_fps * segment_time)
                print(f"  [转换] 保持关键帧间隔: 每 {segment_time} 秒 (GOP≈{gop_size}帧)")
                convert_cmd.extend([
                    '-force_key_frames', f'expr:gte(t,n_forced*{segment_time})',
                    '-g', str(gop_size),  # 设置GOP大小
                ])
            
            convert_cmd.extend([
                '-c:a', 'aac',                  # 音频编码为 AAC
                '-b:a', '192k',
                '-ar', '44100',                 # 统一采样率
                '-ac', '2',                     # 统一为立体声
                output_path
            ])
        
        
        result = subprocess.run(convert_cmd, capture_output=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            
            # 如果 GPU 编码失败，尝试降级到 CPU
            if use_gpu and 'nvenc' in error_msg.lower():
                print(f"  [转换] ⚠ GPU 编码失败，降级到 CPU 编码...")
                convert_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-ar', '44100',
                    '-ac', '2',
                    output_path
                ]
                
                result = subprocess.run(convert_cmd, capture_output=True)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    raise Exception(f"视频转换失败: {error_msg[:500]}")
            else:
                raise Exception(f"视频转换失败: {error_msg[:500]}")
    


# --------------------------------------------------------------------------
# 批量音频提取合并节点
# --------------------------------------------------------------------------
class BatchAudioExtractAndMerge:
    """批量提取视频音频并按顺序合并"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "待提取音频目录": ("STRING", {"default": "output/长视频转换分割", "multiline": False, "placeholder": "包含视频文件的目录", "tooltip": "包含待提取音频视频文件的目录路径。支持相对路径（基于ComfyUI根目录）和绝对路径。目录中的视频文件将按文件名中的数字顺序进行处理。支持多种视频格式如MP4、AVI、MOV、MKV等。程序会自动按文件名中的数字进行智能排序，确保音频提取的顺序正确。"}),
                "合并后音频保存目录": ("STRING", {"default": "output/音频保存", "multiline": False, "placeholder": "合并后音频的保存目录", "tooltip": "音频提取和合并后的保存目录。程序会自动创建目录（如果不存在）。建议使用与输入目录不同的目录，避免文件混乱。支持相对路径和绝对路径，支持中文目录名。目录路径将作为输出信息返回，可连接到其他节点使用。"}),
                "合并后音频文件名": ("STRING", {"default": "merged_audio", "multiline": False, "placeholder": "合并后的音频文件名（不含扩展名）", "tooltip": "合并后音频文件的文件名（不含扩展名）。程序会根据音频处理模式自动添加适当扩展名（如.aac或.mp3）。建议使用简洁明了的名称，避免特殊字符。文件名最终会包含扩展名，例如merged_audio.aac或merged_audio.mp3。"}),
                "音频处理模式": (["流复制(保持原格式)", "转换为AAC"], {"default": "转换为AAC", "description": "流复制速度快但需要格式一致，转换为AAC兼容性好", "tooltip": "音频提取处理模式。流复制(保持原格式): 直接复制音频流，速度最快但需要所有视频音频编码格式一致。转换为AAC: 将所有音频转换为AAC格式，兼容性好，推荐使用（默认设置）。如果视频音频格式不统一，建议选择转换为AAC模式以避免兼容性问题。"}),
                "自动合并提取的音频": ("BOOLEAN", {"default": True, "description": "开启后自动合并所有提取的音频，关闭则只提取不合并", "tooltip": "音频自动合并控制开关。开启后(默认): 自动按顺序合并所有提取的音频为一个文件，适合需要完整音频轨道的场景。关闭后: 只提取音频到输出目录但不合并，适合需要分别处理每个音频文件的场景。对于大量视频文件，关闭此选项可以避免内存占用过高。"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("合并音频路径", "提取音频数量", "输出目录")
    FUNCTION = "extract_and_merge_audio"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "批量提取目录中视频的音频，可选择流复制或转换为AAC格式，支持自动按顺序合并所有提取的音频。开启自动合并开关后会将所有音频合并为一个文件，关闭则只提取音频到输出目录。"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 每次都执行，不使用缓存
        return float("NaN")

    def extract_and_merge_audio(self, 待提取音频目录, 合并后音频保存目录, 合并后音频文件名, 音频处理模式, 自动合并提取的音频):
        try:
            print(f"\n{'='*60}")
            print(f"[批量音频提取] 开始处理...")
            print(f"[批量音频提取] 待提取音频目录: {待提取音频目录}")
            print(f"[批量音频提取] 合并后音频保存目录: {合并后音频保存目录}")
            print(f"[批量音频提取] 合并后音频文件名: {合并后音频文件名}")
            print(f"[批量音频提取] 音频处理模式: {音频处理模式}")
            print(f"[批量音频提取] 自动合并提取的音频: {自动合并提取的音频}")
            
            # 解析路径（相对路径基于ComfyUI根目录）
            待提取音频目录 = resolve_path(待提取音频目录)
            合并后音频保存目录 = resolve_path(合并后音频保存目录)
            
            print(f"[批量音频提取] 解析后待提取路径: {待提取音频目录}")
            print(f"[批量音频提取] 解析后保存路径: {合并后音频保存目录}")
            
            # 检查输入目录是否存在
            input_dir = Path(待提取音频目录)
            if not input_dir.exists() or not input_dir.is_dir():
                print(f"[批量音频提取] ✗ 错误: 输入目录不存在: {待提取音频目录}")
                print(f"{'='*60}\n")
                return ("", 0, 合并后音频保存目录)
            
            # 创建输出目录
            output_dir = Path(合并后音频保存目录)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取所有视频文件（支持多种格式）
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv', '.ts', '.mts', '.m2ts', '.vob']
            all_videos = []
            for ext in video_extensions:
                all_videos.extend(input_dir.glob(f'*{ext}'))
            
            # 按文件名排序（数字排序）
            def numeric_sort_key(item):
                numbers = re.findall(r'\d+', item.name)
                return tuple(map(int, numbers)) if numbers else (float('inf'),)
            
            all_videos = sorted(all_videos, key=numeric_sort_key)
            
            video_count = len(all_videos)
            print(f"[批量音频提取] 找到 {video_count} 个视频文件")
            
            if video_count == 0:
                print(f"[批量音频提取] ⚠️ 目录中没有视频文件")
                print(f"{'='*60}\n")
                return ("", 0, 合并后音频保存目录)
            
            # 显示视频列表
            for i, v in enumerate(all_videos, 1):
                file_size = v.stat().st_size / 1024 / 1024
                print(f"  视频 {i}. {v.name} ({file_size:.2f} MB)")
            
            # 提取音频
            print(f"\n[批量音频提取] 开始提取音频...")
            extracted_audio_files = []
            
            # 确定音频格式
            use_stream_copy = (音频处理模式 == "流复制(保持原格式)")
            
            for i, video_path in enumerate(all_videos, 1):
                print(f"\n[批量音频提取] 处理 {i}/{video_count}: {video_path.name}")
                
                try:
                    # 首先获取视频时长
                    duration_probe_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'json',
                        str(video_path)
                    ]
                    
                    duration_result = subprocess.run(duration_probe_cmd, capture_output=True, text=True)
                    video_duration = 0.0
                    
                    if duration_result.returncode == 0:
                        duration_info = json.loads(duration_result.stdout)
                        video_duration = float(duration_info.get('format', {}).get('duration', 0))
                    
                    # 检测音频流信息
                    probe_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-select_streams', 'a:0',
                        '-show_entries', 'stream=codec_name',
                        '-of', 'json',
                        str(video_path)
                    ]
                    
                    result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    
                    has_audio = False
                    if result.returncode == 0:
                        audio_info = json.loads(result.stdout)
                        if audio_info.get('streams'):
                            has_audio = True
                    
                    # 如果没有音频流，生成静音音频
                    if not has_audio:
                        print(f"  [音频] 视频无音频流，生成 {video_duration:.2f} 秒静音音频以保持时间线一致")
                        
                        if video_duration <= 0:
                            print(f"  [音频] ⚠️ 无法获取视频时长，跳过")
                            continue
                        
                        # 生成静音音频文件
                        output_audio = output_dir / f"{i:04d}.aac"
                        
                        # 使用 FFmpeg 生成静音音频
                        silence_cmd = [
                            'ffmpeg',
                            '-y',
                            '-f', 'lavfi',
                            '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
                            '-t', str(video_duration),  # 设置时长
                            '-acodec', 'aac',
                            '-b:a', '192k',
                            '-ar', '44100',
                            '-ac', '2',
                            str(output_audio)
                        ]
                        
                        print(f"  [音频] 生成静音音频...")
                        result = subprocess.run(silence_cmd, capture_output=True)
                        
                        if result.returncode != 0:
                            error_msg = result.stderr.decode('utf-8', errors='ignore')
                            print(f"  [音频] ✗ 生成静音音频失败: {error_msg[:200]}")
                            continue
                        
                        # 检查输出文件
                        if not output_audio.exists():
                            print(f"  [音频] ✗ 静音音频文件未创建")
                            continue
                        
                        file_size = output_audio.stat().st_size / 1024 / 1024
                        print(f"  [音频] ✓ 静音音频生成成功: {output_audio.name} ({file_size:.2f} MB, {video_duration:.2f}秒)")
                        
                        extracted_audio_files.append(output_audio)
                        continue
                    
                    audio_codec = audio_info['streams'][0].get('codec_name', 'unknown')
                    print(f"  [音频] 音频编码: {audio_codec}")
                    
                    # 生成输出文件名（使用4位数字序号）
                    if use_stream_copy:
                        # 流复制模式：根据原始编码确定扩展名
                        if audio_codec == 'aac':
                            audio_ext = '.aac'
                        elif audio_codec == 'mp3':
                            audio_ext = '.mp3'
                        elif audio_codec == 'opus':
                            audio_ext = '.opus'
                        elif audio_codec == 'vorbis':
                            audio_ext = '.ogg'
                        elif audio_codec == 'flac':
                            audio_ext = '.flac'
                        else:
                            # 未知格式，使用通用扩展名
                            audio_ext = '.audio'
                        
                        output_audio = output_dir / f"{i:04d}{audio_ext}"
                        
                        print(f"  [音频] 流复制模式（{audio_codec}）...")
                        extract_cmd = [
                            'ffmpeg',
                            '-y',
                            '-i', str(video_path),
                            '-vn',                  # 不处理视频
                            '-acodec', 'copy',      # 音频流复制
                            str(output_audio)
                        ]
                    else:
                        # 转换为AAC模式
                        output_audio = output_dir / f"{i:04d}.aac"
                        
                        print(f"  [音频] 转换为AAC格式...")
                        extract_cmd = [
                            'ffmpeg',
                            '-y',
                            '-i', str(video_path),
                            '-vn',                  # 不处理视频
                            '-acodec', 'aac',       # 转换为AAC
                            '-b:a', '192k',         # 比特率192kbps
                            '-ar', '44100',         # 采样率44.1kHz
                            '-ac', '2',             # 立体声
                            str(output_audio)
                        ]
                    
                    # 执行提取
                    result = subprocess.run(extract_cmd, capture_output=True)
                    
                    if result.returncode != 0:
                        error_msg = result.stderr.decode('utf-8', errors='ignore')
                        print(f"  [音频] ✗ 提取失败: {error_msg[:200]}")
                        continue
                    
                    # 检查输出文件
                    if not output_audio.exists():
                        print(f"  [音频] ✗ 输出文件未创建")
                        continue
                    
                    file_size = output_audio.stat().st_size / 1024 / 1024
                    print(f"  [音频] ✓ 提取成功: {output_audio.name} ({file_size:.2f} MB)")
                    
                    extracted_audio_files.append(output_audio)
                    
                except Exception as e:
                    print(f"  [音频] ✗ 处理失败: {str(e)}")
                    continue
            
            extracted_count = len(extracted_audio_files)
            print(f"\n[批量音频提取] ✓ 成功提取 {extracted_count}/{video_count} 个音频文件")
            
            if extracted_count == 0:
                print(f"[批量音频提取] ⚠️ 没有成功提取任何音频")
                print(f"{'='*60}\n")
                return ("", 0, 合并后音频保存目录)
            
            # 如果不自动合并，直接返回
            if not 自动合并提取的音频:
                print(f"[批量音频提取] 自动合并已关闭，音频文件已保存到: {合并后音频保存目录}")
                print(f"{'='*60}\n")
                return ("", extracted_count, 合并后音频保存目录)
            
            # 合并音频
            print(f"\n[批量音频提取] 开始合并音频...")
            
            # 确定合并后的文件扩展名
            if use_stream_copy:
                # 流复制模式：检查所有音频格式是否一致
                audio_formats = set([f.suffix for f in extracted_audio_files])
                if len(audio_formats) == 1:
                    # 格式一致，使用相同扩展名
                    merged_ext = list(audio_formats)[0]
                    print(f"  [合并] 所有音频格式一致: {merged_ext}")
                else:
                    # 格式不一致，需要转换为AAC
                    print(f"  [合并] 音频格式不一致，将转换为AAC格式")
                    merged_ext = '.aac'
                    use_stream_copy = False  # 强制转换模式
            else:
                merged_ext = '.aac'
            
            # 生成合并后的文件名
            merged_filename = f"{合并后音频文件名}{merged_ext}"
            merged_audio_path = output_dir / merged_filename
            
            # 如果文件已存在，添加序号
            if merged_audio_path.exists():
                base_name = 合并后音频文件名
                counter = 1
                while True:
                    merged_filename = f"{base_name}_{counter:03d}{merged_ext}"
                    merged_audio_path = output_dir / merged_filename
                    if not merged_audio_path.exists():
                        break
                    counter += 1
                    if counter > 999:
                        import time
                        timestamp = int(time.time())
                        merged_filename = f"{base_name}_{timestamp}{merged_ext}"
                        merged_audio_path = output_dir / merged_filename
                        break
                print(f"  [合并] 检测到同名文件，自动重命名为: {merged_filename}")
            
            # 创建临时文件列表
            filelist_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            
            try:
                # 写入文件列表
                for audio_file in extracted_audio_files:
                    abs_path = os.path.abspath(str(audio_file)).replace('\\', '/')
                    filelist_file.write(f"file '{abs_path}'\n")
                
                filelist_file.close()
                filelist_path = filelist_file.name
                
                print(f"  [合并] 临时文件列表: {filelist_path}")
                print(f"  [合并] 音频列表:")
                for i, audio in enumerate(extracted_audio_files, 1):
                    print(f"    {i}. {audio.name}")
                
                # 合并音频
                if use_stream_copy:
                    # 流复制模式（快速）
                    print(f"  [合并] 使用流复制模式（快速合并）...")
                    merge_cmd = [
                        'ffmpeg',
                        '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', filelist_path,
                        '-c', 'copy',
                        str(merged_audio_path)
                    ]
                else:
                    # 转换模式（兼容性好）
                    print(f"  [合并] 使用转换模式（统一为AAC格式）...")
                    merge_cmd = [
                        'ffmpeg',
                        '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', filelist_path,
                        '-acodec', 'aac',
                        '-b:a', '192k',
                        '-ar', '44100',
                        '-ac', '2',
                        str(merged_audio_path)
                    ]
                
                print(f"  [合并] 执行合并...")
                result = subprocess.run(merge_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=600)
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    print(f"  [合并] ✗ FFmpeg 错误: {error_msg[:500]}")
                    raise ValueError(f"音频合并失败: {error_msg[:500]}")
                
                # 检查输出文件
                if not merged_audio_path.exists():
                    raise ValueError(f"输出文件未创建: {merged_audio_path}")
                
                file_size = merged_audio_path.stat().st_size / 1024 / 1024
                if file_size == 0:
                    raise ValueError(f"输出文件为空: {merged_audio_path}")
                
                print(f"  [合并] ✓ 合并成功: {merged_audio_path.name}")
                print(f"  [合并] 文件大小: {file_size:.2f} MB")
                
                # 清理临时音频文件（可选）
                print(f"\n[批量音频提取] 清理临时音频文件...")
                for audio_file in extracted_audio_files:
                    try:
                        os.remove(audio_file)
                        print(f"  - 已删除: {audio_file.name}")
                    except Exception as e:
                        print(f"  - 删除失败: {audio_file.name} - {e}")
                
            finally:
                # 清理临时文件列表
                if os.path.exists(filelist_path):
                    try:
                        os.remove(filelist_path)
                    except:
                        pass
            
            print(f"\n[批量音频提取] ✓ 完成!")
            print(f"[批量音频提取] 合并音频路径: {merged_audio_path}")
            print(f"[批量音频提取] 提取音频数量: {extracted_count}")
            print(f"{'='*60}\n")
            
            return (str(merged_audio_path), extracted_count, 合并后音频保存目录)
            
        except Exception as e:
            print(f"[批量音频提取] ✗ 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return ("", 0, 合并后音频保存目录)


# --------------------------------------------------------------------------
# 节点注册
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "zyf_VideoDirectoryLoader": VideoDirectoryLoader,
    "zyf_VideoAutoCombine": VideoAutoCombine,
    "zyf_VideoConvertAndSplit": VideoConvertAndSplit,
    "zyf_BatchAudioExtractAndMerge": BatchAudioExtractAndMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zyf_VideoDirectoryLoader": "视频目录加载器",
    "zyf_VideoAutoCombine": "视频自动合并器",
    "zyf_VideoConvertAndSplit": "视频转换分割器",
    "zyf_BatchAudioExtractAndMerge": "批量音频提取合并器",
}

NODE_DESCRIPTION_MAPPINGS = {
    "zyf_VideoDirectoryLoader": "从指定目录加载视频并拆分为帧序列，支持音频提取和批量处理",
    "zyf_VideoAutoCombine": "自动将新视频追加到已存在的视频文件末尾，实现视频自动合并",
    "zyf_VideoConvertAndSplit": "自动转换视频格式为MP4并按指定秒数分割视频",
    "zyf_BatchAudioExtractAndMerge": "批量提取目录中视频的音频并按顺序合并，支持流复制或转换为AAC",
}
