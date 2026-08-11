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

# 复用图像目录节点的自动排队/状态消息发送函数
from .image_directory_nodes import _send_directory_auto_queue, _send_directory_status, _zyf_natural_key

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
# 强制尺寸目标计算 (与 zyf-video 的 zyf_target_size 逻辑一致)
# --------------------------------------------------------------------------
def _zyf_target_size(width, height, force_size, custom_short_edge=512, custom_long_edge=512, custom_width=480, custom_height=480, size_multiple=0):
    """计算强制尺寸后的目标 (width, height)。"""
    if force_size == "自定义宽高":
        if custom_width > 0 and custom_height > 0:
            target_w = custom_width
            target_h = custom_height
        elif custom_width > 0:
            target_w = custom_width
            target_h = max(1, (height * custom_width) // width)
        elif custom_height > 0:
            target_h = custom_height
            target_w = max(1, (width * custom_height) // height)
        else:
            return (width, height)
    elif force_size == "自定义短边":
        if width < height:
            target_w = custom_short_edge
            target_h = max(1, (height * custom_short_edge) // width)
        else:
            target_h = custom_short_edge
            target_w = max(1, (width * custom_short_edge) // height)
    elif force_size == "自定义长边":
        if width < height:
            target_h = custom_long_edge
            target_w = max(1, (width * custom_long_edge) // height)
        else:
            target_w = custom_long_edge
            target_h = max(1, (height * custom_long_edge) // width)
    elif force_size not in ("禁用", "Disabled"):
        parts = force_size.split("x")
        if parts[0] == "?":
            target_w = (width * int(parts[1])) // height
            target_w = int(target_w) + 4 & ~7
            target_h = int(parts[1])
        elif parts[1] == "?":
            target_h = (height * int(parts[0])) // width
            target_h = int(target_h) + 4 & ~7
            target_w = int(parts[0])
        else:
            target_w = int(parts[0])
            target_h = int(parts[1])
    else:
        return (width, height)

    if size_multiple > 0:
        target_w = max(size_multiple, (target_w // size_multiple) * size_multiple)
        target_h = max(size_multiple, (target_h // size_multiple) * size_multiple)

    return (target_w, target_h)


# --------------------------------------------------------------------------
# GPU 解码检测 (NVDEC)
# --------------------------------------------------------------------------
_gpu_decode_checked = None
_gpu_decode_available = False

def _check_gpu_decode():
    """检测 ffmpeg 是否支持 CUDA hwaccel (NVDEC)。"""
    global _gpu_decode_checked, _gpu_decode_available
    if _gpu_decode_checked is not None:
        return _gpu_decode_available
    _gpu_decode_checked = True
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        _gpu_decode_available = "cuda" in result.stdout.lower()
    except Exception:
        _gpu_decode_available = False
    if _gpu_decode_available:
        print("[ZYF] GPU 硬件解码 (NVDEC) 可用")
    return _gpu_decode_available


# --------------------------------------------------------------------------
# GPU 编码检测 (NVENC)
# --------------------------------------------------------------------------
_gpu_encode_checked = None
_gpu_encode_available = False

def _check_gpu_encode():
    """检测 ffmpeg 是否支持 h264_nvenc/hevc_nvenc 编码器。"""
    global _gpu_encode_checked, _gpu_encode_available
    if _gpu_encode_checked is not None:
        return _gpu_encode_available
    _gpu_encode_checked = True
    try:
        for encoder in ["h264_nvenc", "hevc_nvenc"]:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            if encoder not in result.stdout:
                _gpu_encode_available = False
                return False
        _gpu_encode_available = True
    except Exception:
        _gpu_encode_available = False
    if _gpu_encode_available:
        print("[ZYF] GPU 硬件编码 (NVENC) 可用")
    return _gpu_encode_available


# --------------------------------------------------------------------------
# 视频帧加载函数 (参考 zyf-video 优化：GPU解码 + ffmpeg缩放)
# --------------------------------------------------------------------------
def load_video_frames(video_path, force_size="禁用", custom_short_edge=512, custom_long_edge=832, custom_width=480, custom_height=480, size_multiple=0):
    """使用 FFmpeg 加载视频并提取所有帧、音频和元数据。
    
    Args:
        video_path: 视频文件路径
        force_size: 强制尺寸选项 ("禁用" / "自定义短边" / "自定义长边" / "自定义宽高" / "480x?" / "?x480" / ...)
        custom_short_edge: 自定义短边像素值
        custom_long_edge: 自定义长边像素值
        custom_width: 自定义宽度像素值
        custom_height: 自定义高度像素值
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
            return None, None, 0, 0, 0, 0
        
        stream = video_info['streams'][0]
        
        # 解析帧率
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_parts = fps_str.split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
        
        if abs(fps - 29.97) < 0.01:
            print(f"  [加载] 检测到 NTSC 帧率: {fps:.6f} fps")
        elif abs(fps - 30.0) < 0.01:
            print(f"  [加载] 检测到标准 30 fps")
        else:
            print(f"  [加载] 帧率: {fps:.6f} fps")
        
        # 获取原始尺寸
        orig_width = int(stream.get('width', 0))
        orig_height = int(stream.get('height', 0))
        
        # 计算目标尺寸（强制尺寸逻辑）
        if force_size != "禁用":
            target_w, target_h = _zyf_target_size(
                orig_width, orig_height, force_size,
                custom_short_edge, custom_long_edge, custom_width, custom_height, size_multiple
            )
            print(f"  [加载] 强制尺寸: {force_size} -> {target_w}x{target_h} (原始: {orig_width}x{orig_height})")
        else:
            # 禁用时保持原始尺寸，但规范化为偶数（编码兼容）
            target_w = orig_width - (orig_width % 2)
            target_h = orig_height - (orig_height % 2)
            if target_w != orig_width or target_h != orig_height:
                print(f"  [加载] 原始尺寸: {orig_width}x{orig_height} -> 规范化: {target_w}x{target_h}")
            else:
                print(f"  [加载] 尺寸: {orig_width}x{orig_height}")
        
        # 获取总帧数
        total_frames = int(stream.get('nb_frames', 0))
        if total_frames == 0:
            duration = float(stream.get('duration', 0))
            if duration > 0:
                total_frames = int(duration * fps)
        
        # 使用 FFmpeg 提取所有帧（GPU解码 + ffmpeg缩放）
        decode_start = time.time()
        video_cmd = ['ffmpeg', '-v', 'error']
        
        # GPU 硬件加速解码 (NVDEC)
        use_gpu_decode = _check_gpu_decode()
        if use_gpu_decode:
            video_cmd += ['-hwaccel', 'cuda', '-extra_hw_frames', '3']
            print(f"  [加载] GPU 硬件解码 (NVDEC) 已启用")
        
        video_cmd += ['-i', str(video_path)]
        
        # 缩放滤镜 (ffmpeg Lanczos = SIMD加速，比PIL快得多)
        if force_size != "禁用" and (target_w != orig_width or target_h != orig_height):
            video_cmd += ['-vf', f'scale={target_w}:{target_h}:flags=lanczos']
        
        video_cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        
        print(f"  [加载] 开始解码视频帧...")
        
        def _run_decode(cmd):
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc.communicate()
        
        video_data, stderr = _run_decode(video_cmd)
        
        # GPU 解码失败时自动回退到 CPU
        if use_gpu_decode and len(video_data) == 0:
            err_str = stderr.decode('utf-8', errors='ignore').lower()
            if any(kw in err_str for kw in ("cuda", "hwaccel", "cuvid", "no capable", "failed")):
                print(f"  [加载] GPU 解码失败，自动回退到 CPU 解码")
                cpu_cmd = ['ffmpeg', '-v', 'error', '-i', str(video_path)]
                if force_size != "禁用" and (target_w != orig_width or target_h != orig_height):
                    cpu_cmd += ['-vf', f'scale={target_w}:{target_h}:flags=lanczos']
                cpu_cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
                video_data, stderr = _run_decode(cpu_cmd)
        
        decode_time = time.time() - decode_start
        print(f"  [加载] 解码视频帧耗时: {decode_time:.2f}秒，数据大小: {len(video_data) / 1024 / 1024:.2f} MB")
        
        if len(video_data) == 0:
            print(f"FFmpeg 提取帧失败: {stderr.decode('utf-8', errors='ignore')[:200]}")
            return None, None, 0, 0, orig_width, orig_height
        
        # 转换为 numpy 数组
        convert_start = time.time()
        frame_data = np.frombuffer(video_data, dtype=np.uint8)
        
        bytes_per_frame = target_w * target_h * 3
        actual_frames = len(frame_data) // bytes_per_frame
        
        if actual_frames == 0:
            print(f"未能提取任何帧: {video_path}")
            return None, None, 0, 0, orig_width, orig_height
        
        frame_data = frame_data[:actual_frames * bytes_per_frame]
        frames = frame_data.reshape((actual_frames, target_h, target_w, 3))
        frames = frames.astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames)
        
        convert_time = time.time() - convert_start
        print(f"  [加载] 转换为tensor耗时: {convert_time:.2f}秒")
        
        # 提取音频
        audio_start = time.time()
        audio_dict = None
        try:
            audio_cmd = [
                'ffmpeg',
                '-v', 'error',
                '-i', str(video_path),
                '-vn',
                '-f', 'f32le',
                '-'
            ]
            
            audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_data_bytes, audio_stderr = audio_process.communicate()
            
            if audio_process.returncode != 0:
                raise subprocess.CalledProcessError(audio_process.returncode, audio_cmd)
            
            match = re.search(r', (\d+) Hz, (\w+), ', audio_stderr.decode('utf-8', errors='ignore'))
            
            if match:
                sample_rate = int(match.group(1))
                channels_str = match.group(2)
                if 'stereo' in channels_str:
                    channels = 2
                elif 'mono' in channels_str:
                    channels = 1
                else:
                    channels = 2
            else:
                sample_rate = 44100
                channels = 2
            
            if len(audio_data_bytes) > 0:
                audio_data = torch.frombuffer(bytearray(audio_data_bytes), dtype=torch.float32)
                if len(audio_data) > 0:
                    audio_data = audio_data.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
                    audio_dict = {'waveform': audio_data, 'sample_rate': sample_rate}
                    print(f"  [加载] 提取音频耗时: {time.time() - audio_start:.2f}秒")
                else:
                    audio_dict = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
            else:
                audio_dict = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
                
        except subprocess.CalledProcessError:
            print(f"视频无音频轨道或音频提取失败: {video_path}")
            audio_dict = lambda: {'waveform': torch.zeros((1, 2, 0), dtype=torch.float32), 'sample_rate': 44100}
        
        total_time = time.time() - total_start
        print(f"  [加载] 总耗时: {total_time:.2f}秒")
        
        return frames_tensor, audio_dict, fps, actual_frames, target_w, target_h
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 处理失败: {str(e)}")
        return None, None, 0, 0, 0, 0
    except Exception as e:
        print(f"视频加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0, 0, 0

# --------------------------------------------------------------------------
# 视频目录加载器节点
# --------------------------------------------------------------------------
class VideoDirectoryLoader:
    force_size_options = [
        "禁用",
        "自定义短边",
        "自定义长边",
        "自定义宽高",
        "480x?",
        "?x480",
        "480x480",
        "832x?",
        "?x832",
        "832x832",
    ]

    size_multiple_options = ["无", "8", "16", "32", "64", "128", "256", "512"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "目录路径": ("STRING", {"default": "", "multiline": False, "placeholder": "支持目录或单个视频文件路径", "tooltip": "视频目录或单个视频文件的路径。支持目录（批量加载视频）和单个文件两种模式。支持相对路径（基于ComfyUI根目录）和绝对路径。"}),
                "起始索引": ("INT", {"default": 0, "min": 0, "step": 1, "description": "从第几个视频开始（0表示第1个）", "tooltip": "视频加载的起始位置（0表示第1个视频）。配合单视频顺序加载模式实现批量处理，每次运行自动加载下一个视频。每次重新运行时始终从该索引开始，中断后再次运行也会从头开始，除非手动修改此值。"}),
                "排序方法": (["按名称", "按数字", "按修改时间"], {"default": "按名称", "description": "视频排序方式", "tooltip": "视频排序方式。按名称：按文件名字母顺序；按数字：按文件名中的数字排序（适合序列文件）；按修改时间：按文件最后修改时间排序。"}),
                "递归搜索子目录": ("BOOLEAN", {"default": True, "description": "是否递归查找所有子文件夹", "tooltip": "是否递归搜索所有子目录中的视频文件。开启后将在目录及其所有子目录中查找视频文件；关闭则仅搜索当前目录。"}),
                "文件扩展名过滤": ("STRING", {"default": "", "placeholder": "用逗号分隔，如: mp4,avi,mkv", "description": "留空则加载所有支持的视频格式", "tooltip": "文件扩展名过滤器。用逗号分隔多个扩展名，如'mp4,avi,mkv'。留空则加载所有支持的视频格式。帮助限制仅加载指定格式的视频文件。"}),
                "加载失败跳过": ("BOOLEAN", {"default": True, "description": "加载失败时是否跳过", "tooltip": "当视频加载失败时是否自动跳过。开启后跳过损坏或不支持的文件继续处理；关闭后遇到加载失败将停止处理。建议开启以确保批量处理稳定性。"}),
                "强制尺寸": (VideoDirectoryLoader.force_size_options, {
                    "tooltip": "把视频帧强制缩放到指定尺寸:\n"
                               "  - 禁用         : 保持原始尺寸\n"
                               "  - 自定义短边    : 按短边缩放,短边对齐目标值,长边等比\n"
                               "  - 自定义长边    : 按长边缩放,长边对齐目标值,短边等比\n"
                               "  - 自定义宽高    : 直接指定宽/高,填0则该边等比缩放\n"
                               "  - 480x? / ?x480 : 宽/高对齐 480,另一边等比\n"
                               "  - 832x? / ?x832 : 宽/高对齐 832,另一边等比\n"
                               "  - 480x480 / 832x832 : 固定正方形",
                }),
                "自定义短边": ("INT", {
                    "default": 512, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "自定义短边目标值(像素)。仅当'强制尺寸'为'自定义短边'时生效。",
                }),
                "自定义长边": ("INT", {
                    "default": 832, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "自定义长边目标值(像素)。仅当'强制尺寸'为'自定义长边'时生效。",
                }),
                "自定义宽度": ("INT", {
                    "default": 480, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "自定义宽度目标值(像素)。仅当'强制尺寸'为'自定义宽高'时生效。填 0 表示宽度等比缩放。",
                }),
                "自定义高度": ("INT", {
                    "default": 480, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "自定义高度目标值(像素)。仅当'强制尺寸'为'自定义宽高'时生效。填 0 表示高度等比缩放。",
                }),
                "图像尺寸倍数": (VideoDirectoryLoader.size_multiple_options, {
                    "tooltip": "强制尺寸后的最终尺寸对齐到该倍数。\n"
                               "  - 无 : 不对齐,直接使用计算结果\n"
                               "  - 8/16/32/64/128/256/512 : 把宽高向下取整到该值倍数\n"
                               "仅当'强制尺寸'不为'禁用'时生效。",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "自动索引": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "自动索引（内部用）：由前端自动递增并排队加载下一个视频，无需手动设置。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "ZYF_DIR_VIDEO_INFO", "STRING")
    RETURN_NAMES = ("帧序列", "音频", "视频信息", "相对路径")
    FUNCTION = "load_video"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "从指定目录或单个视频文件加载视频并拆分为帧序列，支持音频提取、强制尺寸缩放、GPU硬件解码、递归搜索、排序等功能。单视频顺序加载模式配合队列实现批量处理。"

    def load_video(self, 目录路径, 起始索引, 排序方法, 递归搜索子目录, 文件扩展名过滤, 加载失败跳过, 强制尺寸, 自定义短边, 自定义长边, 自定义宽度, 自定义高度, 图像尺寸倍数, 自动索引=None, prompt=None, unique_id=None):
        """
        从指定目录或单个视频文件加载视频并拆分为帧序列，支持音频提取、强制尺寸缩放、GPU硬件解码。
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
                _send_directory_status("zyf-video-status", unique_id, 0, 0)
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")
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
            _send_directory_status("zyf-video-status", unique_id, 0, 0)
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")
        
        total_available = len(video_paths)
        
        scan_time = time.time() - start_time
        print(f"[性能] 扫描目录耗时: {scan_time:.2f}秒，找到 {total_available} 个视频")
        
        if total_available == 0:
            print("未找到任何视频文件")
            _send_directory_status("zyf-video-status", unique_id, 0, 0)
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")

        # 排序（优化性能）
        sort_start = time.time()
        if 排序方法 == "按名称":
            # 与图像加载器/ Directory Opus 一致的"自然排序"(忽略大小写, 数字按数值)
            video_paths.sort(key=lambda x: _zyf_natural_key(x.name))
        elif 排序方法 == "按数字":
            # 按文件名中的数字排序
            def numeric_sort_key(item):
                numbers = re.findall(r'\d+', item.name)
                return tuple(map(int, numbers)) if numbers else (float('inf'),)
            video_paths.sort(key=numeric_sort_key)
        elif 排序方法 == "按修改时间":
            # 按修改时间排序（较慢，需要读取文件元数据）
            try:
                video_paths.sort(key=lambda x: x.stat().st_mtime)
            except Exception as e:
                print(f"警告: 按修改时间排序失败，改用按名称排序: {e}")
                video_paths.sort(key=lambda x: _zyf_natural_key(x.name))
        
        sort_time = time.time() - sort_start
        print(f"[性能] 排序耗时: {sort_time:.2f}秒 (方法: {排序方法})")

        # 确定起始索引 —— 由前端缓存的 prompt 中的"自动索引"控件控制，
        # 与 zyf-video 分段计划同款逻辑：无需在执行队列写目录总数，
        # 前端收到自动排队消息后递增索引并自动排队加载下一个视频。
        if 单视频顺序加载:
            try:
                start = int(自动索引) if 自动索引 is not None else 起始索引
            except (TypeError, ValueError):
                start = 起始索引
        else:
            start = 起始索引

        # 通知前端更新顶部状态显示 (已加载数量 / 总数量)
        _send_directory_status("zyf-video-status", unique_id, total_available, start)

        # 单视频模式只加载一个
        if start >= total_available:
            if 单视频顺序加载:
                print(f"✓ 所有视频已处理完成，跳过执行")
                print(f"  - 总视频数: {total_available}")
                print(f"  - 当前索引: {start}")
                print(f"  - 目录路径: {目录路径}")
                print(f"💡 提示: 如需重新处理，请修改目录路径或起始索引")
                # 返回空数据，静默跳过
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")
            else:
                print(f"起始索引 {start} 超出范围，可用视频数: {total_available}")
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")
        
        # 直接使用索引（现在索引从 0 开始）
        video_path = video_paths[start]
        
        # 加载视频 (GPU硬件解码 + ffmpeg Lanczos缩放)
        load_start = time.time()
        frames, audio, fps, total_frames, actual_width, actual_height = load_video_frames(
            str(video_path), 强制尺寸, 自定义短边, 自定义长边, 自定义宽度, 自定义高度, 0 if 图像尺寸倍数 == "无" else int(图像尺寸倍数)
        )
        load_time = time.time() - load_start
        print(f"[性能] 加载视频耗时: {load_time:.2f}秒")
        
        if frames is None:
            if 加载失败跳过:
                print(f"跳过加载失败的视频: {video_path}")
                # 加载失败且允许跳过时，自动排队加载下一个视频
                if start + 1 < total_available:
                    _send_directory_auto_queue("zyf-video-auto-queue", unique_id, start + 1)
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")
            else:
                print(f"加载失败: {video_path}")
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), empty_audio(), {"frame_rate": 0.0, "total_frames": 0, "width": 0, "height": 0}, "")
        
        # 更新索引（当前索引即此视频的索引）
        if 单视频顺序加载:
            # 计算剩余未处理数量（不包括当前这个，因为当前这个正在处理）
            remaining = max(0, total_available - start - 1)
            
            # 智能提示（显示为 1-based 索引更友好）
            display_index = start + 1
            if 智能队列建议 and remaining > 0:
                print(f"▶ 当前索引: {start}  (第 {display_index}/{total_available} 个)  文件: {video_path.name}")
                print(f"  - 帧数: {total_frames}, 帧率: {fps:.2f} fps")
                print(f"  - 剩余未处理: {remaining} 个视频")
                print(f"💡 自动排队: 已自动排队加载下一个视频")
            else:
                print(f"▶ 当前索引: {start}  (第 {display_index}/{total_available} 个)  文件: {video_path.name}")
                print(f"  - 帧数: {total_frames}, 帧率: {fps:.2f} fps")
                if remaining == 0:
                    print(f"✓ 这是最后一个视频")

            # 仍有剩余视频时，通知前端自动排队加载下一个
            if remaining > 0:
                _send_directory_auto_queue("zyf-video-auto-queue", unique_id, start + 1)
        else:
            remaining = 0
        
        # 计算相对路径 (与图像加载器一致, 供视频保存器保持目录结构)
        try:
            相对路径 = str(video_path.relative_to(目录路径))
        except Exception:
            相对路径 = video_path.name
        
        video_info = {"frame_rate": fps, "total_frames": total_frames, "width": actual_width, "height": actual_height}
        return (frames, audio, video_info, 相对路径)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 单视频顺序加载模式默认开启：返回 NaN 确保每次都执行
        # 原因：需要每次检查索引状态，判断是否已完成
        # 如果已完成，会在 load_video 方法中静默跳过
        return float("NaN")

# --------------------------------------------------------------------------
# ffmetadata 辅助函数 (将 ComfyUI 工作流等大体积元数据写入 ffmpeg 格式文件)
# 避免用 -metadata CLI 参数传递超大 JSON 导致 Windows 命令行长度限制错误。
# --------------------------------------------------------------------------
def _write_ffmetadata_file(path, metadata):
    if not metadata:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(";FFMETADATA1\n")
        return

    lines = [";FFMETADATA1"]
    for k, v in metadata.items():
        if v is None:
            continue
        try:
            k_s = str(k)
            v_s = str(v)
        except Exception:
            continue
        v_s = v_s.replace("\\", "\\\\")
        v_s = v_s.replace("\r", "\\r")
        v_s = v_s.replace("\n", "\\n")
        lines.append(f"{k_s}={v_s}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")

# --------------------------------------------------------------------------
# 视频目录保存器节点
# --------------------------------------------------------------------------
class VideoDirectorySaver:
    """将帧序列编码为视频保存到指定目录，支持H264/H265、可选音频和元数据。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "要保存为视频的帧序列张量，形状为(B, H, W, C)，B为帧数。"}),
                "帧率": ("FLOAT", {"default": 32, "step": 1, "tooltip": "输出视频的帧率，支持浮点数，如 23.976、29.97、32 等。默认 32。"}),
                "输出路径": ("STRING", {"default": "output", "multiline": False, "placeholder": "保存到哪个目录", "tooltip": "视频文件的保存目录路径。支持相对路径（基于ComfyUI根目录）和绝对路径。若连接相对路径端口，将保持原始目录结构；否则自动生成名称。"}),
                "编码格式": (["H264", "H265"], {"default": "H264", "tooltip": "视频编码格式。H264：兼容性最好，文件较大；H265(HEVC)：压缩率更高，文件更小，但兼容性略差。"}),
                "质量CRF": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1, "tooltip": "编码质量控制值(CRF)。数值越低质量越高、体积越大。H264 默认 19，H265 默认 22。范围 0-51。"}),
                "保存元数据": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后将 ComfyUI 工作流信息(prompt、workflow)嵌入到视频文件中。当视频拖入 ComfyUI 画布时，会自动还原当时的工作流。关闭则仅保存纯视频。"}),
            },
            "optional": {
                "音频": ("AUDIO", {"default": None, "tooltip": "可选的音频输入。连接后会将音频合并到输出视频中；不连接则输出无音频视频。"}),
                "相对路径": ("STRING", {"forceInput": True, "tooltip": "可选的相对路径输入端口。连接后使用该路径作为输出文件名（保持原始目录结构）；不连接则自动生成默认文件名。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "将帧序列编码为视频保存到指定目录，支持H264/H265编码、可选音频、元数据写入。像素格式固定为yuv420p。"

    def save_video(self, 图像, 帧率, 输出路径, 编码格式, 质量CRF, 保存元数据, 音频=None, 相对路径=None, prompt=None, extra_pnginfo=None):
        """
        将帧序列编码为视频保存到指定目录。支持 GPU 硬件编码 (NVENC) 和逐帧流式写入。
        """
        import hashlib

        # 解析输出目录
        output_dir = Path(resolve_path(输出路径)) if 输出路径.strip() else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 规范化帧序列（保持torch格式，避免整批转numpy）
        if 图像.ndim == 4 and 图像.shape[0] == 1:
            图像 = 图像[0]
        if 图像.ndim == 3:
            图像 = 图像.unsqueeze(0)
        N, H, W, C = 图像.shape
        if C == 1:
            # 单通道转RGB
            图像 = 图像.repeat(1, 1, 1, 3)
            C = 3
        elif C != 3:
            图像 = 图像[..., :3]
            C = 3

        if N == 0 or H == 0 or W == 0:
            print("错误: 图像为空或尺寸无效，无法保存视频")
            return ()

        # 确定输出文件名
        if 相对路径 and str(相对路径).strip():
            rel = str(相对路径).strip().lstrip("/\\")
            output_file = output_dir / rel
            if output_file.suffix.lower() not in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"):
                output_file = output_file.with_suffix(".mp4")
        else:
            frames_np_sample = (图像[0:1].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            content_hash = hashlib.md5(np.ascontiguousarray(frames_np_sample).tobytes()).hexdigest()[:10]
            output_file = output_dir / f"video_{content_hash}.mp4"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            print(f"跳过已存在文件: {output_file}")
            return ()

        # ---- 选择编码器 (GPU NVENC 优先，自动回退 CPU) ----
        use_gpu = _check_gpu_encode()
        if 编码格式 == "H265":
            codec_name = "hevc_nvenc" if use_gpu else "libx265"
        else:
            codec_name = "h264_nvenc" if use_gpu else "libx264"
        print(f"  [编码] 编码器: {codec_name} (GPU={'是' if use_gpu else '否'})")

        # 处理音频
        has_audio = False
        audio_file = None
        if 音频 is not None:
            try:
                if callable(音频):
                    音频 = 音频()
                if isinstance(音频, dict) and 'waveform' in 音频:
                    waveform = 音频['waveform']
                    sample_rate = 音频.get('sample_rate', 44100)
                    if waveform.numel() > 0:
                        if waveform.ndim == 3:
                            waveform = waveform.squeeze(0)
                        if waveform.shape[0] < waveform.shape[1]:
                            waveform = waveform.transpose(0, 1)
                        audio_np = waveform.cpu().numpy().astype(np.float32)
                        audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        audio_cmd = [
                            'ffmpeg', '-y', '-v', 'error',
                            '-f', 'f32le', '-ar', str(sample_rate), '-ac', str(audio_np.shape[1]),
                            '-i', '-', '-c:a', 'pcm_s16le', audio_file.name,
                        ]
                        audio_process = subprocess.Popen(audio_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        try:
                            stdout, stderr = audio_process.communicate(input=audio_np.tobytes(), timeout=60)
                        except subprocess.TimeoutExpired:
                            audio_process.kill()
                            stdout, stderr = audio_process.communicate()
                        if audio_process.returncode == 0:
                            has_audio = True
                        else:
                            audio_file.close()
                            os.remove(audio_file.name)
                            audio_file = None
            except Exception as e:
                print(f"音频处理失败，将保存无音频视频: {e}")
                if audio_file:
                    try:
                        audio_file.close()
                        os.remove(audio_file.name)
                    except:
                        pass
                audio_file = None
                has_audio = False

        meta_file = None
        try:
            # -- 构建元数据 payload (嵌入 ComfyUI 工作流信息) ----------
            if 保存元数据:
                from datetime import datetime, timezone
                metadata_payload = {
                    "title": f"zyf视频目录保存 ({编码格式})",
                    "creation_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                    "encoder": f"zyf-{codec_name}",
                    "zyf_format": 编码格式,
                    "zyf_codec": codec_name,
                    "zyf_pix_fmt": "yuv420p",
                    "zyf_crf": str(质量CRF),
                    "zyf_frame_rate": str(帧率),
                    "zyf_width": str(W),
                    "zyf_height": str(H),
                    "zyf_total_frames": str(N),
                    "zyf_has_audio": "1" if has_audio else "0",
                    "zyf_type": "output",
                }
                if isinstance(prompt, dict):
                    try:
                        metadata_payload["prompt"] = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
                    except Exception:
                        pass
                if isinstance(extra_pnginfo, dict):
                    try:
                        for k, v in extra_pnginfo.items():
                            metadata_payload[str(k)] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                    except Exception:
                        pass

                meta_file = tempfile.NamedTemporaryFile(suffix='.ffmeta', delete=False, dir=output_dir)
                _write_ffmetadata_file(meta_file.name, metadata_payload)

            # -- 构建 FFmpeg 命令 ------------------------------------
            def _build_cmd(use_codec):
                cmd = [
                    'ffmpeg', '-y', '-v', 'error',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', f'{W}x{H}', '-r', f'{帧率:.6f}',
                    '-i', '-',
                ]
                if has_audio and audio_file:
                    cmd += ['-i', audio_file.name]

                nv = use_codec in ("h264_nvenc", "hevc_nvenc")
                if nv:
                    cmd += [
                        '-c:v', use_codec,
                        '-preset', 'p4',
                        '-tune', 'hq',
                        '-pix_fmt', 'yuv420p',
                        '-cq', str(int(质量CRF)),
                        '-rc', 'vbr',
                    ]
                else:
                    cmd += [
                        '-c:v', use_codec,
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-crf', str(int(质量CRF)),
                    ]

                if has_audio:
                    cmd += ['-c:a', 'aac', '-b:a', '192k', '-shortest']

                # 偶数尺寸修正
                even_w, even_h = W, H
                if W % 2 != 0 or H % 2 != 0:
                    even_w = W + (W % 2)
                    even_h = H + (H % 2)
                    cmd += ['-vf', f'pad={even_w}:{even_h}:0:0:black']

                if meta_file:
                    cmd += ['-map', '0:v:0']
                    if has_audio and audio_file:
                        cmd += ['-map', '1:a:0?']
                    meta_input_index = 2 if (has_audio and audio_file) else 1
                    cmd += ['-i', meta_file.name, '-map_metadata', str(meta_input_index)]
                    cmd += ['-movflags', '+faststart+use_metadata_tags']
                else:
                    cmd += ['-movflags', '+faststart']

                cmd.append(str(output_file))
                return cmd

            def _run_encode(use_codec):
                cmd = _build_cmd(use_codec)
                print(f"  [编码] 开始编码 {N} 帧...")
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # 逐帧流式写入 (避免一次性分配整批uint8数组)
                try:
                    progress_step = max(1, N // 10)
                    for i in range(N):
                        frame = 图像[i].cpu().numpy()
                        frame = (np.clip(frame, 0.0, 1.0) * 255).clip(0, 255).astype(np.uint8)
                        frame_bytes = frame.tobytes()
                        try:
                            proc.stdin.write(frame_bytes)
                        except BrokenPipeError:
                            break
                        del frame_bytes
                        if (i + 1) % progress_step == 0 or i == N - 1:
                            pct = int((i + 1) * 100 / N)
                            print(f"  [编码] 进度: {i + 1}/{N} 帧 ({pct}%)")
                    proc.stdin.close()
                except Exception:
                    try:
                        proc.kill()
                    except:
                        pass
                    raise
                proc.wait()
                stderr_text = proc.stderr.read().decode('utf-8', errors='ignore') if proc.stderr else ''
                return proc.returncode, stderr_text

            rc, stderr = _run_encode(codec_name)

            # GPU 编码失败时自动回退到 CPU
            if rc != 0 and use_gpu:
                err_str = stderr.lower()
                if any(kw in err_str for kw in ("cuda", "nvenc", "no capable", "driver")):
                    cpu_codec = "libx265" if 编码格式 == "H265" else "libx264"
                    print(f"  [编码] GPU 编码失败，自动回退到 CPU 编码器 {cpu_codec}")
                    try:
                        if os.path.isfile(str(output_file)):
                            os.remove(str(output_file))
                    except OSError:
                        pass
                    rc, stderr = _run_encode(cpu_codec)
                    if rc == 0:
                        codec_name = cpu_codec

            if rc != 0:
                raise Exception(f"FFmpeg 编码失败: {stderr[:500]}")

            if not os.path.exists(str(output_file)) or os.path.getsize(str(output_file)) == 0:
                raise Exception(f"输出文件异常: {output_file}")

            file_size = os.path.getsize(str(output_file))
            print(f"  [编码] ✓ 视频已保存: {output_file}")
            print(f"  [编码] 帧率: {帧率}, 编码: {编码格式}, 编码器: {codec_name}, CRF: {质量CRF}, 大小: {file_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"错误: 保存视频失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if meta_file:
                try:
                    meta_file.close()
                    os.remove(meta_file.name)
                except:
                    pass
            if audio_file:
                try:
                    audio_file.close()
                    os.remove(audio_file.name)
                except:
                    pass

        return ()


# --------------------------------------------------------------------------
# 视频信息解包节点
# 将视频目录加载器输出的视频信息字典解包为帧率、总帧数、宽度、高度。
# --------------------------------------------------------------------------
class VideoInfoUnpack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "视频信息": ("ZYF_DIR_VIDEO_INFO", {"tooltip": "从视频目录加载器输出的视频信息包，包含帧率、总帧数、宽度、高度。"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("帧率", "总帧数", "宽度", "高度")
    FUNCTION = "unpack"
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "将视频目录加载器输出的视频信息字典解包为帧率、总帧数、宽度、高度四个独立端口。"

    def unpack(self, 视频信息):
        if not isinstance(视频信息, dict):
            return (0.0, 0, 0, 0)
        return (
            视频信息.get("frame_rate", 0.0),
            视频信息.get("total_frames", 0),
            视频信息.get("width", 0),
            视频信息.get("height", 0),
        )


# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "zyf_VideoDirectoryLoader": VideoDirectoryLoader,
    "zyf_VideoDirectorySaver": VideoDirectorySaver,
    "zyf_VideoInfoUnpack": VideoInfoUnpack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zyf_VideoDirectoryLoader": "视频目录加载器",
    "zyf_VideoDirectorySaver": "视频目录保存器",
    "zyf_VideoInfoUnpack": "视频信息解包",
}

NODE_DESCRIPTION_MAPPINGS = {
    "zyf_VideoDirectoryLoader": "从指定目录加载视频并拆分为帧序列，支持音频提取、强制尺寸缩放、GPU硬件解码、递归搜索和批量处理",
    "zyf_VideoDirectorySaver": "将帧序列编码为视频保存到指定目录，支持H264/H265编码、GPU硬件编码(NVENC)、可选音频和元数据写入",
    "zyf_VideoInfoUnpack": "将视频目录加载器输出的视频信息字典解包为帧率、总帧数、宽度、高度四个独立端口",
}
