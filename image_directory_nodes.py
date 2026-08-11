import os
import re
import torch
from PIL import Image, ImageOps, ImageSequence
import numpy as np
from pathlib import Path
import json
import hashlib
import time
import threading
import folder_paths
from comfy.utils import common_upscale

# --------------------------------------------------------------------------
# 自然排序键 (与 Directory Opus / Windows 资源管理器默认"按名称(正序)"一致)
# 忽略大小写, 并把名称中的数字按数值比较(如 2 < 10), 而非按字符比较。
# --------------------------------------------------------------------------
def _zyf_natural_key(name):
    return tuple(
        (0, int(chunk), '') if chunk.isdigit() else (1, 0, chunk.casefold())
        for chunk in re.split(r'(\d+)', name)
        if chunk != ''
    )

# --------------------------------------------------------------------------
# 自定义图像加载函数 (兼容 ComfyUI 标准格式)
# --------------------------------------------------------------------------
def load_image(image_path):
    """加载图像并转换为 ComfyUI 兼容的张量格式 (batch, height, width, channels)"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            # 保证 shape 为 (H, W, 3)
            if img_np.ndim == 2:  # 灰度图
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[-1] != 3:
                img_np = img_np[..., :3]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            return img_tensor
    except Exception as e:
        print(f"图像加载失败: {str(e)}")
        return None

# --------------------------------------------------------------------------
# 自动排队消息发送 (与 zyf-video 分段计划同款机制)
# 通知前端递增"自动索引"并自动排队加载下一个文件, 无需在执行队列写目录总数。
# --------------------------------------------------------------------------
def _send_directory_auto_queue(message_type, unique_id, next_index):
    """通知前端自动排队加载下一个文件。"""
    if not unique_id:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            message_type,
            {"uid": unique_id, "next_index": next_index},
        )
    except Exception as e:
        print(f"[zyf_image_directory] 自动排队消息发送失败: {e}")

# --------------------------------------------------------------------------
# 状态消息发送
# 通知前端更新插件顶部的状态显示 (已加载数量 / 总数量), 实时显示处理进度。
# --------------------------------------------------------------------------
def _send_directory_status(message_type, unique_id, total, loaded):
    """通知前端更新顶部状态显示: 已加载数量 / 总数量。"""
    if not unique_id:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            message_type,
            {"uid": unique_id, "total": total, "loaded": loaded},
        )
    except Exception as e:
        print(f"[zyf_image_directory] 状态消息发送失败: {e}")

# --------------------------------------------------------------------------
# 目录文件计数 (供前端添加/改变目录后即时显示总数)
# kind: 'image' / 'video'; 复用文件列表缓存, 与节点运行扫描逻辑一致。
# --------------------------------------------------------------------------
def count_directory_files(kind, 目录路径, 递归搜索子目录=True, 文件扩展名过滤="", 排序方法="按名称"):
    """统计目录中匹配的文件总数 (kind: 'image' / 'video')。"""
    if not 目录路径 or not os.path.exists(目录路径):
        return 0
    path = Path(目录路径)
    if path.is_file():
        return 1
    if kind == "image":
        if not os.path.isdir(目录路径):
            return 0
        return len(ImageDirectoryLoader._get_cached_file_list(
            目录路径, 递归搜索子目录, 文件扩展名过滤, 排序方法
        ))
    # video
    if not os.path.isdir(目录路径):
        return 0
    if 文件扩展名过滤.strip():
        video_extensions = tuple(f".{ext.strip().lower()}" for ext in 文件扩展名过滤.split(",") if ext.strip())
    else:
        video_extensions = (
            ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
            ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts", ".mts",
            ".m2ts", ".vob", ".rm", ".rmvb", ".asf", ".divx"
        )
    paths = set()
    if 递归搜索子目录:
        for ext in video_extensions:
            paths.update(path.rglob(f'*{ext}'))
    else:
        for ext in video_extensions:
            paths.update(path.glob(f'*{ext}'))
    return len(paths)


def is_valid_image_to_save(img_tensor):
    """
    检测图像是否为空图像或占位图像，不应该保存
    返回: (is_valid, reason)
    - is_valid: True表示可以保存，False表示不应保存
    - reason: 不保存的原因
    """
    try:
        # 处理批次维度
        if img_tensor.ndim == 4:
            if img_tensor.shape[0] == 0:
                return False, "空批次图像"
            # 取第一张图像进行检测
            img = img_tensor[0]
        else:
            img = img_tensor
        
        # 检查图像尺寸
        if img.ndim < 2:
            return False, "无效的图像维度"
        
        height, width = img.shape[0], img.shape[1]
        
        # 检测占位图像 (64x64 或更小的纯色图像)
        if height <= 64 and width <= 64:
            # 检查是否为纯色图像（所有像素值相同或接近）
            img_np = img.cpu().numpy()
            if img_np.size == 0:
                return False, "空图像数据"
            
            # 计算标准差，纯色图像标准差接近0
            std = np.std(img_np)
            if std < 0.001:  # 标准差阈值
                mean_val = np.mean(img_np)
                # 检查是否为全黑或全白
                if mean_val < 0.01:
                    return False, f"占位图像 (全黑 {width}x{height})"
                elif mean_val > 0.99:
                    return False, f"占位图像 (全白 {width}x{height})"
                else:
                    return False, f"占位图像 (纯色 {width}x{height})"
        
        # 检查是否为全黑图像（任意尺寸）
        img_np = img.cpu().numpy()
        if np.max(img_np) < 0.01:
            return False, f"全黑图像 ({width}x{height})"
        
        # 通过所有检测，图像有效
        return True, ""
        
    except Exception as e:
        print(f"图像验证时出错: {str(e)}")
        # 出错时保守处理，允许保存
        return True, ""

# --------------------------------------------------------------------------
# 图像目录加载器节点 (性能优化版)
# --------------------------------------------------------------------------
class ImageDirectoryLoader:
    _auto_index = {}  # 内存缓存的自动索引
    _file_list_cache = {}  # 文件列表缓存 {cache_key: (file_list, timestamp)}
    _cache_dirty = False  # 标记缓存是否需要写入
    _last_save_time = 0  # 上次保存时间
    _save_lock = threading.Lock()  # 线程锁
    _index_loaded = False  # 标记是否已加载索引
    SAVE_INTERVAL = 3.0  # 每3秒最多保存一次
    FILE_CACHE_TTL = 300  # 文件列表缓存5分钟
    
    @classmethod
    def _get_cache_file(cls):
        """获取缓存文件路径"""
        cache_dir = Path(__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "auto_index.json"
    
    @classmethod
    def _load_auto_index(cls):
        """从文件加载自动索引（仅在内存缓存为空时）"""
        if cls._index_loaded:
            return  # 已加载，跳过
        
        with cls._save_lock:
            if cls._index_loaded:  # 双重检查
                return
            
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
            
            cls._index_loaded = True
    
    @classmethod
    def _save_auto_index(cls, force=False):
        """延迟保存自动索引到文件（批量写入优化）"""
        current_time = time.time()
        
        # 如果不是强制保存，且距离上次保存时间不足间隔，标记为脏数据后返回
        if not force and (current_time - cls._last_save_time) < cls.SAVE_INTERVAL:
            cls._cache_dirty = True
            return
        
        if not cls._cache_dirty and not force:
            return
        
        with cls._save_lock:
            try:
                cache_file = cls._get_cache_file()
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cls._auto_index, f, ensure_ascii=False, indent=2)
                cls._cache_dirty = False
                cls._last_save_time = current_time
            except Exception as e:
                print(f"保存自动索引缓存失败: {e}")
    
    @classmethod
    def _get_file_list_cache_key(cls, 目录路径, 递归搜索子目录, 文件扩展名过滤, 排序方法):
        """生成文件列表缓存键"""
        key_str = f"{目录路径}#{递归搜索子目录}#{文件扩展名过滤}#{排序方法}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    @classmethod
    def _get_cached_file_list(cls, 目录路径, 递归搜索子目录, 文件扩展名过滤, 排序方法):
        """获取缓存的文件列表（避免重复扫描目录）"""
        cache_key = cls._get_file_list_cache_key(目录路径, 递归搜索子目录, 文件扩展名过滤, 排序方法)
        
        # 检查缓存是否存在且有效
        if cache_key in cls._file_list_cache:
            file_list, timestamp = cls._file_list_cache[cache_key]
            if time.time() - timestamp < cls.FILE_CACHE_TTL:
                return file_list
        
        # 缓存失效或不存在，重新扫描
        if 文件扩展名过滤.strip():
            image_extensions = tuple(f".{ext.strip().lower()}" for ext in 文件扩展名过滤.split(",") if ext.strip())
        else:
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
        
        image_paths = []
        path = Path(目录路径)
        if 递归搜索子目录:
            for ext in image_extensions:
                image_paths.extend(path.rglob(f'*{ext}'))
        else:
            for ext in image_extensions:
                image_paths.extend(path.glob(f'*{ext}'))
        
        image_paths = list(set(image_paths))
        
        # 排序
        if 排序方法 == "按名称":
            image_paths.sort(key=lambda x: _zyf_natural_key(x.name))
        elif 排序方法 == "按数字":
            def numeric_sort_key(item):
                rel_path = str(item.relative_to(目录路径))
                numbers = re.findall(r'\d+', rel_path)
                return tuple(map(int, numbers)) if numbers else (float('inf'),)
            image_paths.sort(key=numeric_sort_key)
        elif 排序方法 == "按修改时间":
            image_paths.sort(key=lambda x: x.stat().st_mtime)
        
        # 更新缓存
        cls._file_list_cache[cache_key] = (image_paths, time.time())
        
        return image_paths
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "目录路径": ("STRING", {"default": "", "multiline": False, "placeholder": "请输入要加载的图像目录路径", "tooltip": "图像文件所在目录路径，支持相对路径（基于ComfyUI根目录）和绝对路径。将从此目录加载所有符合条件的图像文件。"}),
                "起始索引": ("INT", {"default": 0, "min": 0, "step": 1, "description": "从第几张图片开始（0表示第1张）", "tooltip": "开始加载的图像索引位置（从0开始）。例如设为5表示跳过前5张图像，从第6张开始加载。常用于分批加载或断点续传。每次重新运行时始终从该索引开始，中断后再次运行也会从头开始，除非手动修改此值。"}),
                "排序方法": (["按名称", "按数字", "按修改时间"], {"default": "按名称", "description": "图片排序方式", "tooltip": "图像文件的排序方式。按名称：按文件名字母顺序排序；按数字：按文件名字中的数字排序；按修改时间：按文件的最后修改时间排序。"}),
                "递归搜索子目录": ("BOOLEAN", {"default": True, "description": "是否递归查找所有子文件夹", "tooltip": "是否在子目录中递归搜索图像文件。开启后将从当前目录及其所有子目录中加载图像；关闭则仅加载当前目录下的图像。"}),
                "文件扩展名过滤": ("STRING", {"default": "", "placeholder": "用逗号分隔，如: jpg,png", "description": "留空则加载所有支持的图片格式(jpg,jpeg,png,bmp,webp,tiff)", "tooltip": "要加载的图像文件扩展名列表，用逗号分隔。支持格式：jpg、jpeg、png、bmp、webp、tiff等。留空则加载所有支持的图像格式。"}),
                "加载失败跳过": ("BOOLEAN", {"default": True, "description": "加载失败时是否跳过", "tooltip": "遇到无法读取的图像文件时是否跳过继续处理。开启后将自动跳过损坏或不支持的图像文件；关闭则遇到错误时停止加载。"}),
                "转换为RGBA": ("BOOLEAN", {"default": False, "description": "是否将图像转换为RGBA透明通道格式，启用后将以PNG格式保存", "tooltip": "是否将加载的图像转换为RGBA格式并添加透明通道。开启后图像将保存为PNG格式，透明度为完全不透明（255）。适用于需要透明通道的后续处理。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "自动索引": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "自动索引（内部用）：由前端自动递增并排队加载下一张，无需手动设置。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "STRING")
    RETURN_NAMES = ("图像", "相对路径", "文件名")
    FUNCTION = "load_images"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "从指定目录批量加载图片，支持递归、排序、扩展名过滤等功能。单张顺序加载模式使用持久化缓存，确保每次运行自动加载下一张图片。所有图片处理完成后会自动跳过执行。"

    def load_images(self, 目录路径, 起始索引, 排序方法, 递归搜索子目录, 文件扩展名过滤, 加载失败跳过, 转换为RGBA, 自动索引=None, prompt=None, unique_id=None):
        """
        从指定目录批量加载图片，支持递归、排序、扩展名过滤等功能
        
        Args:
            目录路径 (str): 图像目录路径，支持相对路径（基于ComfyUI根目录）和绝对路径
            起始索引 (int): 从第几张开始加载（0表示第1张），常用于分批加载或断点续传
            排序方法 (str): 排序方式："按名称"、"按数字"、"按修改时间"
            递归搜索子目录 (bool): 是否在子目录中递归搜索图像文件
            文件扩展名过滤 (str): 用逗号分隔的文件扩展名列表，留空则加载所有支持格式
            加载失败跳过 (bool): 遇到无法读取的图像文件时是否自动跳过
            转换为RGBA (bool): 是否将图像转换为RGBA格式并添加透明通道
        
        Returns:
            tuple:
                - 图像 (torch.Tensor): 加载的图像张量，形状为(B, H, W, C)
                - 相对路径 (list): 相对路径列表
                - 文件名 (str): 当前文件名
        
        Notes:
            - 单张顺序加载模式使用持久化缓存，确保每次运行自动加载下一张图片
            - 所有图片处理完成后会自动跳过执行
            - 支持中文路径和多种图像格式：jpg、jpeg、png、bmp、webp、tiff
            - 自动检测并跳过无效图像文件
        """
        # 后台默认开启的选项
        单张顺序加载 = True  # 默认开启单张顺序加载模式
        智能队列建议 = True  # 默认开启智能队列建议
        
        if not os.path.isdir(目录路径):
            print(f"错误: 目录 '{目录路径}' 不存在")
            _send_directory_status("zyf-image-status", unique_id, 0, 0)
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "")

        # 使用缓存的文件列表（性能优化：避免重复扫描目录）
        image_paths = ImageDirectoryLoader._get_cached_file_list(
            目录路径, 递归搜索子目录, 文件扩展名过滤, 排序方法
        )
        
        total_available = len(image_paths)
        if total_available == 0:
            print("未找到任何图像文件")
            _send_directory_status("zyf-image-status", unique_id, 0, 0)
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "")

        # 确定加载数量
        if 单张顺序加载:
            最大加载数量 = 1
        else:
            # 批量模式下加载所有剩余图片
            最大加载数量 = len(image_paths)
            
        # 确定起始索引 —— 由前端缓存的 prompt 中的"自动索引"控件控制，
        # 与 zyf-video 分段计划同款逻辑：无需在执行队列写目录总数，
        # 前端收到自动排队消息后递增索引并自动排队加载下一张。
        if 单张顺序加载:
            try:
                start = int(自动索引) if 自动索引 is not None else 起始索引
            except (TypeError, ValueError):
                start = 起始索引
        else:
            start = 起始索引

        # 通知前端更新顶部状态显示 (已加载数量 / 总数量)
        _send_directory_status("zyf-image-status", unique_id, total_available, start)

        # 检查是否已处理完成（单张顺序加载模式）
        if start >= total_available:
            if 单张顺序加载:
                print(f"✓ 所有图片已处理完成，跳过执行")
                print(f"  - 总图片数: {total_available}")
                print(f"  - 当前索引: {start}")
                print(f"  - 目录路径: {目录路径}")
                print(f"💡 提示: 如需重新处理，请修改目录路径或起始索引")
                # 返回空数据，静默跳过
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "")
            else:
                print(f"未选择任何图像。起始索引 {start} 可能过高")
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "")
        
        # 直接使用索引（现在索引从 0 开始）
        array_index = start
        end = array_index + 最大加载数量
        selected_paths = image_paths[array_index:end]
        total_loaded = len(selected_paths)
        if total_loaded == 0:
            print(f"未选择任何图像。起始索引 {start} 可能过高")
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "")

        images = []
        relative_paths = []
        for img_path in selected_paths:
            img_tensor = load_image(str(img_path))
            if img_tensor is not None:
                # 转换为RGBA格式
                if 转换为RGBA:
                    # 添加alpha通道（全不透明）
                    alpha_channel = torch.ones_like(img_tensor[:, :, :, 0:1])
                    img_tensor = torch.cat([img_tensor, alpha_channel], dim=-1)
                    # 修改文件扩展名为.png
                    original_rel_path = img_path.relative_to(目录路径).as_posix()
                    rel_path = os.path.splitext(original_rel_path)[0] + '.png'
                else:
                    rel_path = img_path.relative_to(目录路径).as_posix()
                images.append(img_tensor)
                relative_paths.append(rel_path)
            elif not 加载失败跳过:
                print(f"加载失败: {img_path}")
                break

        if not images:
            print("未成功加载任何有效图像")
            # 加载失败且允许跳过时，自动排队加载下一张
            if 单张顺序加载 and 加载失败跳过:
                _send_directory_auto_queue("zyf-image-auto-queue", unique_id, start + 1)
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "")

        # 单张顺序加载模式处理
        if 单张顺序加载 and len(images) > 0:
            # 当前索引即此张的索引，剩余未处理 = 总 - 当前 - 1

            # 根据RGBA设置调整文件名
            if selected_paths and 转换为RGBA:
                original_name = selected_paths[0].name
                文件名 = os.path.splitext(original_name)[0] + '.png'
            else:
                文件名 = selected_paths[0].name if selected_paths else ""
            
            # 计算剩余未处理数量（不包括当前这张，因为当前这张正在处理）
            remaining = max(0, total_available - start - 1)
            
            # 智能提示信息（显示为 1-based 索引更友好）
            display_index = start + 1
            if 智能队列建议 and remaining > 0:
                print(f"▶ 当前索引: {start}  (第 {display_index}/{total_available} 张)  文件: {文件名}")
                print(f"💡 自动排队: 已自动排队加载下一张图片")
            else:
                print(f"▶ 当前索引: {start}  (第 {display_index}/{total_available} 张)  文件: {文件名}")
                if remaining == 0:
                    print(f"✓ 这是最后一张图片")

            # 仍有剩余图片时，通知前端自动排队加载下一张
            if remaining > 0:
                _send_directory_auto_queue("zyf-image-auto-queue", unique_id, start + 1)

            return (images[0], [relative_paths[0]], 文件名)
        
        batch_images = torch.cat(images, dim=0)
        return (batch_images, relative_paths, "")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 单张顺序加载模式默认开启：返回 NaN 确保每次都执行
        # 原因：需要每次检查索引状态，判断是否已完成
        # 如果已完成，会在 load_images 方法中静默跳过
        return float("NaN")
    
    def __del__(self):
        """析构函数：确保在对象销毁时保存缓存"""
        ImageDirectoryLoader._save_auto_index(force=True)

# --------------------------------------------------------------------------
# 图像目录保存器节点
# --------------------------------------------------------------------------
class ImageDirectorySaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像批量": ("IMAGE", {"tooltip": "要保存的批量图像数据。可以是单张图像或图像序列，将根据相对路径列表保持原始目录结构进行保存。"}),
                "输出路径": ("STRING", {"default": "output", "multiline": False, "placeholder": "保存到哪个目录", "tooltip": "图像文件的保存目录路径。支持相对路径（基于ComfyUI根目录）和绝对路径。如果目录不存在，将自动创建。"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否", "tooltip": "当目标位置已存在同名文件时是否覆盖。开启后将直接覆盖已存在的文件；关闭则跳过已存在的文件，避免数据丢失。"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式", "tooltip": "图像保存格式。原格式：保持原始图像格式；jpg/webp：标准有损压缩格式，适合照片；png：无损压缩格式，支持透明度，适合图形和截图。"}),
                "压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效", "tooltip": "JPG和WEBP格式的压缩质量（1-100）。数值越高，图像质量越好，文件越大。推荐设置：照片用85-95，网页用70-85，测试用50-70。PNG格式此设置无效。"}),
                "保存元数据": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后保存图片元数据信息（如生成参数、工作流信息等），并自动使用PNG格式保存。关闭则不保存元数据，可使用任意格式。"}),
            },
            "optional": {
                "相对路径列表": ("LIST", {"default": None, "description": "可选：连接时使用原始路径，不连接时使用默认文件名", "tooltip": "可选的相对路径列表输入。如果连接此端口，将优先使用提供的路径列表保存图像，保持原始目录结构；不连接时将使用默认的文件名生成规则。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "批量保存图像到指定目录，保持原始相对路径结构"

    def save_images(self, 图像批量, 输出路径, 覆盖已存在文件, 保存格式, 压缩质量, 保存元数据, 相对路径列表=None, prompt=None, extra_pnginfo=None):
        """
        批量保存图像到指定目录，保持原始相对路径结构
        
        Args:
            图像批量 (torch.Tensor): 要保存的图像张量，形状为(B, H, W, C)或(H, W, C)
            输出路径 (str): 保存目录路径，支持相对路径和绝对路径
            覆盖已存在文件 (bool): 是否覆盖已存在的同名文件
            保存格式 (str): 保存格式，可选值："原格式"、"jpg"、"png"、"webp"
            压缩质量 (int): JPG和WEBP格式的压缩质量（1-100）
            保存元数据 (bool): 是否保存元数据信息
            相对路径列表 (list, optional): 相对路径列表，保持原始目录结构
            prompt (dict, optional): 工作流提示词信息
            extra_pnginfo (dict, optional): 额外的PNG信息，包含完整工作流数据
        
        Returns:
            tuple: 空元组，此节点无返回值
        
        Notes:
            - 支持批量图像和单张图像保存
            - 如果未提供相对路径列表，将自动生成默认文件名
            - 自动检测并跳过无效图像（空图像、占位图像）
            - 自动创建不存在的目录
            - 支持中文路径
        """
        output_dir = Path(输出路径) if 输出路径.strip() else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果开启保存元数据，强制使用PNG格式
        if 保存元数据:
            print("保存元数据已开启，自动使用PNG格式")
            保存格式 = "png"
        
        # 如果没有提供相对路径列表，生成默认文件名
        if 相对路径列表 is None or len(相对路径列表) == 0:
            print("未连接相对路径列表，使用按内容哈希的默认文件名（幂等，不会重复保存）")
            # 确定文件扩展名
            if 保存格式 != "原格式":
                ext = f".{保存格式}"
            else:
                ext = ".png"

            # 按图像内容哈希命名，保证同一张图永远对应同一文件名，
            # 中途重跑/重复加载同一张图时会命中"已存在文件"跳过逻辑，避免内容重复。
            相对路径列表 = []
            for idx in range(len(图像批量)):
                img = 图像批量[idx]
                if img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                content_hash = hashlib.md5(np.ascontiguousarray(img.cpu().numpy()).tobytes()).hexdigest()[:10]
                filename = f"image_{content_hash}{ext}"
                相对路径列表.append(filename)
        
        if len(图像批量) != len(相对路径列表):
            print(f"错误: 图像数量 ({len(图像批量)}) 与路径数量 ({len(相对路径列表)}) 不匹配")
            return ()

        saved_count = 0
        skipped_count = 0
        for idx, (img_tensor, rel_path) in enumerate(zip(图像批量, 相对路径列表)):
            # 验证图像是否有效
            is_valid, reason = is_valid_image_to_save(img_tensor)
            if not is_valid:
                print(f"跳过保存 {rel_path}: {reason}")
                skipped_count += 1
                continue
            
            output_path = output_dir / rel_path
            # 如果是原格式则保留原始扩展名，否则使用指定格式
            if 保存格式 != '原格式':
                output_path = output_path.with_suffix(f'.{保存格式}')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not 覆盖已存在文件:
                print(f"跳过已存在文件: {output_path}")
                skipped_count += 1
                continue
            try:
                if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                    img_tensor = img_tensor[0]
                img_np = img_tensor.cpu().numpy()
                # 保证 shape 为 (H, W, 3)
                if img_np.ndim == 3 and img_np.shape[-1] != 3 and img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                save_kwargs = {}
                if 保存格式 in ["jpg", "webp"]:
                    save_kwargs["quality"] = 压缩质量
                
                # 如果开启保存元数据且使用PNG格式，添加元数据
                if 保存元数据 and 保存格式 == "png":
                    from datetime import datetime
                    from PIL.PngImagePlugin import PngInfo
                    pnginfo = PngInfo()
                    
                    # 基础元数据
                    pnginfo.add_text("Software", "ComfyUI zyf_image_directory_nodes")
                    pnginfo.add_text("Creation Time", datetime.now().isoformat())
                    pnginfo.add_text("Node", "ImageDirectorySaver")
                    pnginfo.add_text("Metadata Enabled", "true")
                    
                    # 工作流元数据
                    if prompt is not None:
                        import json
                        pnginfo.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        import json
                        for key, value in extra_pnginfo.items():
                            pnginfo.add_text(key, json.dumps(value))
                    
                    save_kwargs["pnginfo"] = pnginfo
                    print(f"已添加完整工作流元数据到: {output_path}")
                
                img.save(output_path, **save_kwargs)
                saved_count += 1
                print(f"已保存图像: {output_path}")
            except Exception as e:
                print(f"保存图像失败 {output_path}: {str(e)}")
        
        print(f"保存完成: {saved_count} 张已保存, {skipped_count} 张已跳过")
        return ()

# --------------------------------------------------------------------------
# 图像保存与预览节点
# --------------------------------------------------------------------------
class ImageSaveWithPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "要保存的单张或批量图像数据。支持单张图像保存和批量图像序列保存（如视频帧），会根据设置自动处理并提供预览功能。"}),
                "输出路径": ("STRING", {"default": "output", "placeholder": "保存目录路径，默认为output文件夹", "tooltip": "图像文件的保存目录路径。支持相对路径（基于ComfyUI根目录）和绝对路径。如果目录不存在，将自动创建。默认为output文件夹。"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否", "tooltip": "当目标位置已存在同名文件时是否覆盖。开启后将直接覆盖已存在的文件；关闭则自动生成不重复的文件名（添加数字后缀）避免覆盖。"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式", "tooltip": "图像保存格式。原格式：保持原始图像格式；jpg/webp：标准有损压缩格式，适合照片；png：无损压缩格式，支持透明度，适合图形和截图。"}),
                "压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效", "tooltip": "JPG和WEBP格式的压缩质量（1-100）。数值越高，图像质量越好，文件越大。推荐设置：照片用85-95，网页用70-85，测试用50-70。PNG格式此设置无效。"}),
                "保存元数据": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后保存图片元数据信息（如生成参数、工作流信息等），并自动使用PNG格式保存。关闭则不保存元数据，可使用任意格式。"}),
            },
            "optional": {
                "文件名": ("STRING", {"default": "", "description": "从加载图像节点连接的文件名文本，不连接则使用自动生成的文件名", "tooltip": "从图像加载节点连接的文件名文本。如果连接此端口，将使用提供的前缀名称生成文件；不连接则使用自动生成的默认文件名（如frame_0001.png）。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("预览图像",)
    FUNCTION = "save_and_preview"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "保存图像到指定路径并提供预览功能，支持批量保存视频帧序列和单张图像"

    def save_and_preview(self, 图像, 输出路径, 覆盖已存在文件, 保存格式, 压缩质量, 保存元数据, 文件名="", prompt=None, extra_pnginfo=None):
        """
        保存图像到指定路径并提供预览功能，支持批量保存视频帧序列和单张图像
        
        Args:
            图像 (torch.Tensor): 要保存的图像张量，形状为(B, H, W, C)或(H, W, C)
            输出路径 (str): 保存目录路径，支持相对路径和绝对路径
            覆盖已存在文件 (bool): 是否覆盖已存在的同名文件
            保存格式 (str): 保存格式，可选值："原格式"、"jpg"、"png"、"webp"
            压缩质量 (int): JPG和WEBP格式的压缩质量（1-100）
            保存元数据 (bool): 是否保存元数据信息
            文件名 (str, optional): 从图像加载节点连接的文件名文本
            prompt (dict, optional): 工作流提示词信息
            extra_pnginfo (dict, optional): 额外的PNG信息，包含完整工作流数据
        
        Returns:
            tuple: 包含以下元素的元组
                - 预览图像 (torch.Tensor): 与输入相同的图像张量，用于预览
                - 输出路径 (str): 实际输出路径
                - 保存数量 (int): 成功保存的图像数量
        
        Notes:
            - 支持单张图像和批量图像（视频帧序列）保存
            - 自动检测并跳过无效图像
            - 自动创建不存在的目录
            - 批量保存时自动生成帧序号
        """
        # 验证图像是否有效
        is_valid, reason = is_valid_image_to_save(图像)
        if not is_valid:
            print(f"跳过保存: {reason}")
            return (图像,)
        
        # 处理输出路径
        save_dir = Path(输出路径) if 输出路径.strip() else Path("output")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果开启保存元数据，强制使用PNG格式
        if 保存元数据:
            print("保存元数据已开启，自动使用PNG格式")
            保存格式 = "png"

        # 检查是批量图像还是单张图像
        is_batch = 图像.ndim == 4 and 图像.shape[0] > 1
        
        if is_batch:
            # 批量保存模式（视频帧序列）
            print(f"[批量保存] 检测到 {图像.shape[0]} 帧图像序列")
            saved_count = 0
            saved_paths = []
            
            # 确定扩展名
            if 保存格式 != "原格式":
                ext = f".{保存格式}"
            else:
                ext = ".png"
            
            # 如果有文件名，使用它作为基础名
            if 文件名:
                import re
                def clean_filename(name):
                    name = name.replace('/', '_').replace('\\', '_')
                    name = re.sub(r'[^\w\.\-]', '_', name)
                    return os.path.splitext(name)[0]  # 移除扩展名
                base = clean_filename(文件名)
            else:
                base = "frame"
            
            # 批量保存所有帧
            for idx in range(图像.shape[0]):
                frame = 图像[idx]
                
                # 验证每一帧是否有效
                frame_valid, frame_reason = is_valid_image_to_save(frame)
                if not frame_valid:
                    print(f"[批量保存] 跳过帧 {idx}: {frame_reason}")
                    continue
                
                filename = f"{base}_{idx:04d}{ext}"
                output_path = save_dir / filename
                
                # 如果文件存在且不覆盖，跳过
                if output_path.exists() and not 覆盖已存在文件:
                    counter = 1
                    while True:
                        new_filename = f"{base}_{idx:04d}_{counter:04d}{ext}"
                        output_path = save_dir / new_filename
                        if not output_path.exists():
                            break
                        counter += 1
                        if counter > 99999:
                            print(f"[批量保存] 跳过帧 {idx}：无法生成唯一文件名（超过99999次尝试）")
                            continue
                
                try:
                    img_np = frame.cpu().numpy()
                    if img_np.ndim == 3 and img_np.shape[-1] != 3 and img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_np)
                    
                    save_kwargs = {}
                    save_format = 保存格式
                    if 保存格式 == "原格式":
                        save_format = 'PNG'
                    else:
                        if 保存格式 in ["jpg", "webp"]:
                            save_kwargs["quality"] = 压缩质量
                        save_format = 'WebP' if 保存格式 == 'webp' else (保存格式.upper() if 保存格式 != 'jpg' else 'JPEG')
                    
                    # 如果开启保存元数据且使用PNG格式，添加元数据
                    if 保存元数据 and save_format == 'PNG':
                        from datetime import datetime
                        from PIL.PngImagePlugin import PngInfo
                        pnginfo = PngInfo()
                        
                        # 基础元数据
                        pnginfo.add_text("Software", "ComfyUI zyf_image_directory_nodes")
                        pnginfo.add_text("Creation Time", datetime.now().isoformat())
                        pnginfo.add_text("Node", "ImageSaveWithPreview")
                        pnginfo.add_text("Metadata Enabled", "true")
                        
                        # 工作流元数据
                        if prompt is not None:
                            import json
                            pnginfo.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            import json
                            for key, value in extra_pnginfo.items():
                                pnginfo.add_text(key, json.dumps(value))
                        
                        save_kwargs["pnginfo"] = pnginfo
                        print(f"[批量保存] 已添加完整工作流元数据到: {output_path}")
                    
                    img.save(output_path, format=save_format, **save_kwargs)
                    saved_count += 1
                    saved_paths.append(str(output_path))
                except Exception as e:
                    print(f"[批量保存] 保存帧 {idx} 失败: {str(e)}")
            
            print(f"[批量保存] 成功保存 {saved_count}/{图像.shape[0]} 帧到: {save_dir}")
            # 返回第一帧作为预览
            return (图像[0:1],)
        
        else:
            # 单张图像保存模式
            # 处理文件名和扩展名
            if not 文件名:
                # 生成默认序号文件名
                base = "image"
                # 确定扩展名
                if 保存格式 != "原格式":
                    ext = f".{保存格式}"
                else:
                    ext = ".png"  # 原格式默认使用png
                counter = 1
                while True:
                    # 使用4位数字格式，支持最多9999张图片
                    filename = f"{base}_{counter:04d}{ext}"
                    output_path = save_dir / filename
                    if not output_path.exists():
                        break
                    counter += 1
                    if counter > 99999:
                        raise Exception("超过最大尝试次数（99999），无法生成唯一文件名")
            else:
                # 清理文件名，移除非法字符和路径分隔符
                import re
                def clean_filename(name):
                    # 替换路径分隔符为下划线
                    name = name.replace('/', '_').replace('\\', '_')
                    # 移除非字母数字、点、下划线、连字符的字符
                    name = re.sub(r'[^\w\.\-]', '_', name)
                    return name
                
                filename = clean_filename(文件名)
                
                # 处理保存格式
                if 保存格式 != "原格式":
                    # 移除现有扩展名
                    filename = os.path.splitext(filename)[0] + f".{保存格式}"
                
                output_path = save_dir / filename
                # 确保父目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查文件是否存在
        if output_path.exists() and not 覆盖已存在文件:
            # 生成带序号的新文件名
            base = output_path.stem
            ext = output_path.suffix
            counter = 1
            while True:
                new_filename = f"{base}_{counter:04d}{ext}"
                new_output_path = output_path.parent / new_filename
                if not new_output_path.exists():
                    output_path = new_output_path
                    break
                counter += 1
                if counter > 99999:
                    raise Exception("超过最大尝试次数（99999），无法生成唯一文件名")

        try:
            # 处理图像张量
            if 图像.ndim == 4 and 图像.shape[0] == 1:
                img_tensor = 图像[0]
            else:
                img_tensor = 图像
            img_np = img_tensor.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[-1] != 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            # 保存图像
            save_kwargs = {}
            # 处理原格式保存
            save_format = 保存格式
            if 保存格式 == "原格式":
                ext = output_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    save_format = 'JPEG'
                    save_kwargs["quality"] = 压缩质量
                elif ext == '.webp':
                    save_format = 'WebP'
                    save_kwargs["quality"] = 压缩质量
                elif ext == '.png':
                    save_format = 'PNG'
                else:
                    save_format = 'PNG'  # 默认格式
            else:
                if 保存格式 in ["jpg", "webp"]:
                    save_kwargs["quality"] = 压缩质量
                save_format = 'WebP' if 保存格式 == 'webp' else (保存格式.upper() if 保存格式 != 'jpg' else 'JPEG')
            # 如果文件名没有扩展名，则根据保存格式添加
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{save_format.lower()}")
            
            # 如果开启保存元数据且使用PNG格式，添加元数据
            if 保存元数据 and save_format == 'PNG':
                from datetime import datetime
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                
                # 基础元数据
                pnginfo.add_text("Software", "ComfyUI zyf_image_directory_nodes")
                pnginfo.add_text("Creation Time", datetime.now().isoformat())
                pnginfo.add_text("Node", "ImageSaveWithPreview")
                pnginfo.add_text("Metadata Enabled", "true")
                
                # 工作流元数据
                if prompt is not None:
                    import json
                    pnginfo.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    import json
                    for key, value in extra_pnginfo.items():
                        pnginfo.add_text(key, json.dumps(value))
                
                save_kwargs["pnginfo"] = pnginfo
                print(f"已添加完整工作流元数据到: {output_path}")
            
            img.save(output_path, format=save_format, **save_kwargs)
            print(f"图像已保存: {output_path}")
            return (图像,)
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return (图像,)


# --------------------------------------------------------------------------
# 图像批量合并节点（支持空输入）
# --------------------------------------------------------------------------
class ImageBatchMulti:
    """
    将多个图像合并为一个批量图像。
    与KJNodes的ImageBatchMulti类似，但允许部分输入为空。
    只要至少有一个实际图像输入就正常处理，空输入自动跳过。
    所有输入图像会被缩放到相同尺寸（以第一个有效图像为准）。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "目录加载与保存"
    DESCRIPTION = """
将多个图像合并为一个批量图像。
支持动态输入数量（通过 inputcount 设置）。
允许部分输入为空，只要至少有一个实际图像输入就正常处理。
所有图像会被缩放到相同尺寸。
"""
    
    def combine(self, inputcount, **kwargs):
        # 收集所有非空的有效图像输入
        valid_images = []
        for c in range(inputcount):
            img = kwargs.get(f"image_{c + 1}")
            if img is not None:
                # 检查是否为空张量（0 batch）
                if isinstance(img, torch.Tensor) and img.shape[0] > 0:
                    valid_images.append(img)
        
        # 至少需要一个有效图像
        if len(valid_images) == 0:
            raise ValueError("ImageBatchMulti: 至少需要一个有效的图像输入，但所有输入端口均为空！")
        
        # 以第一个有效图像为基准
        first = valid_images[0]
        h, w = first.shape[1], first.shape[2]
        max_ch = first.shape[-1]
        total_frames = first.shape[0]
        
        # 计算总帧数和最大通道数
        for img in valid_images[1:]:
            max_ch = max(max_ch, img.shape[-1])
            total_frames += img.shape[0]
        
        # 预分配输出
        out = torch.empty((total_frames, h, w, max_ch), dtype=first.dtype, device=first.device)
        offset = 0
        
        for img in valid_images:
            # 缩放尺寸不匹配的图像
            if img.shape[1:3] != (h, w):
                img = common_upscale(img.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
            
            # 补通道数到最大
            if img.shape[-1] < max_ch:
                img = torch.nn.functional.pad(img, (0, max_ch - img.shape[-1]), mode='constant', value=1.0)
            
            n = img.shape[0]
            out[offset:offset + n].copy_(img, non_blocking=True)
            offset += n
        
        return (out.cpu(),)


# --------------------------------------------------------------------------
# 遮罩批量合并节点（支持空输入）
# --------------------------------------------------------------------------
class MaskBatchMulti:
    """
    将多个遮罩合并为一个批量遮罩。
    与KJNodes的MaskBatchMulti类似，但允许部分输入为空。
    只要至少有一个实际遮罩输入就正常处理，空输入自动跳过。
    所有遮罩会被缩放到相同尺寸（以第一个有效遮罩为准）。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "mask_1": ("MASK",),
            },
            "optional": {
                "mask_2": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "combine"
    CATEGORY = "目录加载与保存"
    DESCRIPTION = """
将多个遮罩合并为一个批量遮罩。
支持动态输入数量（通过 inputcount 设置）。
允许部分输入为空，只要至少有一个实际遮罩输入就正常处理。
所有遮罩会被缩放到相同尺寸。
"""
    
    def combine(self, inputcount, **kwargs):
        # 收集所有非空的有效遮罩输入
        valid_masks = []
        for c in range(inputcount):
            m = kwargs.get(f"mask_{c + 1}")
            if m is not None:
                if isinstance(m, torch.Tensor) and m.shape[0] > 0:
                    valid_masks.append(m)
        
        # 至少需要一个有效遮罩
        if len(valid_masks) == 0:
            raise ValueError("MaskBatchMulti: 至少需要一个有效的遮罩输入，但所有输入端口均为空！")
        
        # 以第一个有效遮罩为基准
        mask = valid_masks[0]
        
        for new_mask in valid_masks[1:]:
            if mask.shape[1:] != new_mask.shape[1:]:
                new_mask = torch.nn.functional.interpolate(
                    new_mask.unsqueeze(1),
                    size=(mask.shape[1], mask.shape[2]),
                    mode="bicubic"
                ).squeeze(1)
            mask = torch.cat((mask, new_mask), dim=0)
        
        return (mask,)


# --------------------------------------------------------------------------
# 加载图像节点（支持上游图像输入）
# --------------------------------------------------------------------------
class LoadImageWithInput:
    """
    加载图像节点：与ComfyUI内置的加载图像基本一致，但增加可选的上游图像输入端口。
    
    当上游图像端口连接时，优先使用上游图像数据；
    当上游图像端口未连接时，从磁盘加载选择的图像文件。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "上游图像": ("IMAGE", {"default": None, "tooltip": "可选的上游图像输入。连接后优先使用此图像，忽略文件选择。不连接则从磁盘加载选择的图像文件。"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "load_image"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "加载图像节点，与内置加载图像类似但增加上游图像输入端口。连接上游图像时优先使用上游数据，否则从磁盘加载。"
    
    def load_image(self, image, 上游图像=None):
        # 检查上游图像是否有效（非空且有实际数据）
        upstream_valid = (
            上游图像 is not None
            and isinstance(上游图像, torch.Tensor)
            and 上游图像.shape[0] > 0
        )

        if upstream_valid:
            # 上游图像有效，优先使用
            img_tensor = 上游图像
            if img_tensor.shape[-1] == 4:
                # RGBA图像：提取alpha通道作为遮罩
                mask = 1.0 - img_tensor[..., 3]
                img_tensor = img_tensor[..., :3]  # 只保留RGB
            else:
                # 无alpha通道，返回空遮罩
                if img_tensor.ndim == 4:
                    mask = torch.zeros((img_tensor.shape[0], 64, 64), dtype=img_tensor.dtype, device=img_tensor.device)
                else:
                    mask = torch.zeros((64, 64), dtype=img_tensor.dtype, device=img_tensor.device)

            # 将上游图像保存为临时文件，使节点预览区显示上游图像
            preview_files = self._save_upstream_to_temp(img_tensor)
            return {"ui": {"images": preview_files}, "result": (img_tensor, mask)}

        # 上游无效或为空，回退到从磁盘加载图像
        if 上游图像 is not None:
            print("LoadImageWithInput: 上游图像为空，回退到磁盘加载")

        image_path = folder_paths.get_annotated_filepath(image)

        img = Image.open(image_path)
        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda p: p * (1.0 / 255.0)).convert("L")

            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = 1.0 - torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((64, 64), dtype=torch.float32)

            output_images.append(image_tensor)
            output_masks.append(mask_tensor.unsqueeze(0))

        if len(output_images) == 0:
            # 回退：无有效帧时返回空
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), torch.zeros((0, 64, 64), dtype=torch.float32))

        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)

        return (output_image, output_mask)

    def _save_upstream_to_temp(self, img_tensor):
        """将上游图像张量保存为临时PNG文件，返回ui.images格式的文件列表。
        用于在节点预览区显示上游传入的图像。"""
        import time as _time
        import hashlib as _hashlib

        # 准备输出路径
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)

        results = []
        batch_size = img_tensor.shape[0] if img_tensor.ndim == 4 else 1

        # 为每一帧保存为临时PNG
        for idx in range(batch_size):
            frame = img_tensor[idx] if img_tensor.ndim == 4 else img_tensor
            # 转 numpy
            frame_np = frame.cpu().numpy()
            if frame_np.ndim == 3 and frame_np.shape[-1] != 3 and frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)

            # 唯一文件名（含时间戳与哈希避免冲突）
            uniq = _hashlib.md5(f"{_time.time_ns()}_{idx}".encode()).hexdigest()[:8]
            filename = f"zyf_upstream_{uniq}_{idx:04d}.png"
            full_path = os.path.join(output_dir, filename)
            frame_pil.save(full_path, format="PNG")
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp"
            })

        return results
    
    @classmethod
    def IS_CHANGED(s, image, 上游图像=None):
        # 如果有上游图像输入，始终返回NaN确保每次都执行
        if 上游图像 is not None:
            return float("NaN")
        # 否则基于文件内容判断
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def VALIDATE_INPUTS(s, image, 上游图像=None):
        # 如果有上游图像，跳过文件验证
        if 上游图像 is not None:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "无效的图像文件: {}".format(image)
        return True


# --------------------------------------------------------------------------
# 节点注册
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "zyf_ImageDirectoryLoader": ImageDirectoryLoader,
    "zyf_ImageDirectorySaver": ImageDirectorySaver,
    "zyf_ImageSaveWithPreview": ImageSaveWithPreview,
    "zyf_LoadImageWithInput": LoadImageWithInput,
    "zyf_ImageBatchMulti": ImageBatchMulti,
    "zyf_MaskBatchMulti": MaskBatchMulti,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zyf_ImageDirectoryLoader": "图像目录加载器",
    "zyf_ImageDirectorySaver": "图像目录保存器",
    "zyf_ImageSaveWithPreview": "图像保存与预览器",
    "zyf_LoadImageWithInput": "加载图像(支持上游输入)",
    "zyf_ImageBatchMulti": "图像批量合并",
    "zyf_MaskBatchMulti": "遮罩批量合并",

}

NODE_DESCRIPTION_MAPPINGS = {
    "zyf_ImageDirectoryLoader": "从指定目录加载图像，支持多种排序和过滤选项",
    "zyf_ImageDirectorySaver": "将图像批量保存到指定目录，保持原始相对路径结构",
    "zyf_ImageSaveWithPreview": "保存单张图像到指定路径并提供预览，支持连接加载节点的文件名",
    "zyf_LoadImageWithInput": "加载图像，与内置加载图像类似但增加上游图像输入端口。连接上游图像时优先使用上游数据，否则从磁盘加载。",
    "zyf_ImageBatchMulti": "将多个图像合并为一个批量图像。允许部分输入为空，只要至少有一个有效图像输入就正常处理。",
    "zyf_MaskBatchMulti": "将多个遮罩合并为一个批量遮罩。允许部分输入为空，只要至少有一个有效遮罩输入就正常处理。",

}