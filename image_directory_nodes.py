import os
import re
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
import hashlib
import time
import threading

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
# 图像验证函数 (检测空图像和占位图像)
# --------------------------------------------------------------------------
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
    def _get_file_list_cache_key(cls, 目录路径, 递归搜索子目录, 文件扩展名过滤, sort_method):
        """生成文件列表缓存键"""
        key_str = f"{目录路径}#{递归搜索子目录}#{文件扩展名过滤}#{sort_method}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    @classmethod
    def _get_cached_file_list(cls, 目录路径, 递归搜索子目录, 文件扩展名过滤, sort_method):
        """获取缓存的文件列表（避免重复扫描目录）"""
        cache_key = cls._get_file_list_cache_key(目录路径, 递归搜索子目录, 文件扩展名过滤, sort_method)
        
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
        if sort_method == "按名称":
            image_paths.sort(key=lambda x: str(x.relative_to(目录路径)))
        elif sort_method == "按数字":
            def numeric_sort_key(item):
                rel_path = str(item.relative_to(目录路径))
                numbers = re.findall(r'\d+', rel_path)
                return tuple(map(int, numbers)) if numbers else (float('inf'),)
            image_paths.sort(key=numeric_sort_key)
        elif sort_method == "按修改时间":
            image_paths.sort(key=lambda x: x.stat().st_mtime)
        
        # 更新缓存
        cls._file_list_cache[cache_key] = (image_paths, time.time())
        
        return image_paths
    
    @classmethod
    def _get_key(cls, 目录路径, 任务批次编号):
        """生成缓存键"""
        key_str = f"{目录路径}#{任务批次编号}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "目录路径": ("STRING", {"default": "", "multiline": False, "placeholder": "请输入要加载的图像目录路径", "tooltip": "图像文件所在目录路径，支持相对路径（基于ComfyUI根目录）和绝对路径。将从此目录加载所有符合条件的图像文件。"}),
                "起始索引": ("INT", {"default": 0, "min": 0, "step": 1, "description": "从第几张图片开始（0表示第1张）", "tooltip": "开始加载的图像索引位置（从0开始）。例如设为5表示跳过前5张图像，从第6张开始加载。常用于分批加载或断点续传。"}),
                "任务批次编号": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1, "description": "任务批次标识，变化时重置自动索引", "tooltip": "任务批次标识符，用于区分不同的加载任务。当目录路径或批次编号发生变化时，会自动重置加载索引。此功能便于管理和切换不同的处理批次。"}),
                "sort_method": (["按名称", "按数字", "按修改时间"], {"default": "按名称", "description": "图片排序方式", "tooltip": "图像文件的排序方式。按名称：按文件名字母顺序排序；按数字：按文件名字中的数字排序；按修改时间：按文件的最后修改时间排序。"}),
                "递归搜索子目录": ("BOOLEAN", {"default": True, "description": "是否递归查找所有子文件夹", "tooltip": "是否在子目录中递归搜索图像文件。开启后将从当前目录及其所有子目录中加载图像；关闭则仅加载当前目录下的图像。"}),
                "文件扩展名过滤": ("STRING", {"default": "", "placeholder": "用逗号分隔，如: jpg,png", "description": "留空则加载所有支持的图片格式(jpg,jpeg,png,bmp,webp,tiff)", "tooltip": "要加载的图像文件扩展名列表，用逗号分隔。支持格式：jpg、jpeg、png、bmp、webp、tiff等。留空则加载所有支持的图像格式。"}),
                "加载失败跳过": ("BOOLEAN", {"default": True, "description": "加载失败时是否跳过", "tooltip": "遇到无法读取的图像文件时是否跳过继续处理。开启后将自动跳过损坏或不支持的图像文件；关闭则遇到错误时停止加载。"}),
                "转换为RGBA": ("BOOLEAN", {"default": False, "description": "是否将图像转换为RGBA透明通道格式，启用后将以PNG格式保存", "tooltip": "是否将加载的图像转换为RGBA格式并添加透明通道。开启后图像将保存为PNG格式，透明度为完全不透明（255）。适用于需要透明通道的后续处理。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "STRING", "INT", "INT")
    RETURN_NAMES = ("图像", "相对路径", "filename_text", "可用总数", "剩余未处理")
    FUNCTION = "load_images"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "从指定目录批量加载图片，支持递归、排序、扩展名过滤等功能。单张顺序加载模式使用持久化缓存，确保每次运行自动加载下一张图片。所有图片处理完成后会自动跳过执行。"

    def load_images(self, 目录路径, 起始索引, 任务批次编号, sort_method, 递归搜索子目录, 文件扩展名过滤, 加载失败跳过, 转换为RGBA):
        """
        从指定目录批量加载图片，支持递归、排序、扩展名过滤等功能
        
        Args:
            目录路径 (str): 图像目录路径，支持相对路径（基于ComfyUI根目录）和绝对路径
            起始索引 (int): 从第几张开始加载（0表示第1张），常用于分批加载或断点续传
            任务批次编号 (int): 任务批次标识符，用于区分不同的加载任务
            sort_method (str): 排序方式："按名称"、"按数字"、"按修改时间"
            递归搜索子目录 (bool): 是否在子目录中递归搜索图像文件
            文件扩展名过滤 (str): 用逗号分隔的文件扩展名列表，留空则加载所有支持格式
            加载失败跳过 (bool): 遇到无法读取的图像文件时是否自动跳过
            转换为RGBA (bool): 是否将图像转换为RGBA格式并添加透明通道
        
        Returns:
            tuple: 包含以下元素的元组
                - 图像 (torch.Tensor): 加载的图像张量，形状为(B, H, W, C)
                - 相对路径 (list): 相对路径列表
                - filename_text (str): 当前文件名
                - 可用总数 (int): 目录中可用图像总数
                - 剩余未处理 (int): 剩余待处理图像数量
        
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
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", 0, 0)

        # 使用缓存的文件列表（性能优化：避免重复扫描目录）
        image_paths = ImageDirectoryLoader._get_cached_file_list(
            目录路径, 递归搜索子目录, 文件扩展名过滤, sort_method
        )
        
        total_available = len(image_paths)
        if total_available == 0:
            print("未找到任何图像文件")
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", 0, 0)

        # 确定加载数量
        if 单张顺序加载:
            最大加载数量 = 1
        else:
            # 批量模式下加载所有剩余图片
            最大加载数量 = len(image_paths)
            
        # 确定起始索引
        if 单张顺序加载:
            # 加载自动索引缓存
            ImageDirectoryLoader._load_auto_index()
            
            # 生成缓存键
            cache_key = ImageDirectoryLoader._get_key(目录路径, str(任务批次编号))
            
            # 检查是否需要重置（路径或批次号变化）
            last_config_key = "_last_config"
            current_config = f"{目录路径}#{任务批次编号}"
            config_changed = False
            
            if (last_config_key not in ImageDirectoryLoader._auto_index or 
                ImageDirectoryLoader._auto_index[last_config_key] != current_config):
                # 重置当前批次的自动索引
                ImageDirectoryLoader._auto_index[cache_key] = 起始索引
                ImageDirectoryLoader._auto_index[last_config_key] = current_config
                ImageDirectoryLoader._save_auto_index()
                config_changed = True
                print(f"检测到配置变化，已重置自动索引到起始位置: {起始索引}")
            
            # 单张顺序加载模式：自动递增索引
            if cache_key not in ImageDirectoryLoader._auto_index:
                ImageDirectoryLoader._auto_index[cache_key] = 起始索引
                ImageDirectoryLoader._save_auto_index()
            
            # 获取当前索引并递增（为下次运行准备）
            start = ImageDirectoryLoader._auto_index[cache_key]
            next_index = start + 1
            
            # 延迟保存递增后的索引（性能优化：批量写入）
            ImageDirectoryLoader._auto_index[cache_key] = next_index
            # 每100次操作强制保存一次，防止数据丢失
            force_save = (next_index % 100 == 0)
            ImageDirectoryLoader._save_auto_index(force=force_save)
        else:
            start = 起始索引
        
        # 检查是否已处理完成（单张顺序加载模式）
        if start >= total_available:
            if 单张顺序加载:
                # 任务完成，强制保存缓存
                ImageDirectoryLoader._save_auto_index(force=True)
                print(f"✓ 所有图片已处理完成，跳过执行")
                print(f"  - 总图片数: {total_available}")
                print(f"  - 当前索引: {start}")
                print(f"  - 目录路径: {目录路径}")
                print(f"  - 任务批次: {任务批次编号}")
                print(f"💡 提示: 如需重新处理，请修改目录路径或任务批次编号")
                # 返回空数据，静默跳过
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", total_available, 0)
            else:
                print(f"未选择任何图像。起始索引 {start} 可能过高")
                return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", total_available, 0)
        
        # 直接使用索引（现在索引从 0 开始）
        array_index = start
        end = array_index + 最大加载数量
        selected_paths = image_paths[array_index:end]
        total_loaded = len(selected_paths)
        if total_loaded == 0:
            print(f"未选择任何图像。起始索引 {start} 可能过高")
            # 计算剩余未处理数量
            remaining = max(0, total_available - start) if 单张顺序加载 else 0
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", total_available, remaining)

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
            # 计算剩余未处理数量
            remaining = max(0, total_available - start) if 单张顺序加载 else 0
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", total_available, remaining)

        # 单张顺序加载模式处理
        if 单张顺序加载 and len(images) > 0:
            # 注意：索引已经在前面递增并保存了，这里不需要再次更新
            
            # 根据RGBA设置调整文件名
            if selected_paths and 转换为RGBA:
                original_name = selected_paths[0].name
                filename_text = os.path.splitext(original_name)[0] + '.png'
            else:
                filename_text = selected_paths[0].name if selected_paths else ""
            
            # 计算剩余未处理数量（不包括当前这张，因为当前这张正在处理）
            remaining = max(0, total_available - start - 1)
            
            # 智能提示信息（显示为 1-based 索引更友好）
            display_index = start + 1
            if 智能队列建议 and remaining > 0:
                print(f"当前加载: {filename_text} (第 {display_index}/{total_available} 张)")
                print(f"💡 智能建议: 下次从索引 {start + 1} 开始，队列设置为 {remaining} 次可完成剩余图片处理")
            else:
                print(f"当前加载: {filename_text} (第 {display_index}/{total_available} 张)")
                if remaining == 0:
                    print(f"✓ 这是最后一张图片")
            
            return (images[0], [relative_paths[0]], filename_text, total_available, remaining)
        
        batch_images = torch.cat(images, dim=0)
        # 批量加载模式下，剩余未处理数量为0（因为一次性加载了指定数量）
        remaining = 0
        return (batch_images, relative_paths, "", total_available, remaining)
    
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
                "输出目录": ("STRING", {"default": "output", "multiline": False, "placeholder": "保存到哪个目录", "tooltip": "图像文件的保存目录路径。支持相对路径（基于ComfyUI根目录）和绝对路径。如果目录不存在，将自动创建。"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "tooltip": "当目标位置已存在同名文件时是否覆盖。开启后将直接覆盖已存在的文件；关闭则跳过已存在的文件，避免数据丢失。"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式", "tooltip": "图像保存格式。原格式：保持原始图像格式；jpg/webp：标准有损压缩格式，适合照片；png：无损压缩格式，支持透明度，适合图形和截图。"}),
                "JPG_WEBP_压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效", "tooltip": "JPG和WEBP格式的压缩质量（1-100）。数值越高，图像质量越好，文件越大。推荐设置：照片用85-95，网页用70-85，测试用50-70。PNG格式此设置无效。"}),
                "保存元数据": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后保存图片元数据信息（如生成参数、工作流信息等），并自动使用PNG格式保存。关闭则不保存元数据，可使用任意格式。"}),
            },
            "optional": {
                "相对路径列表": ("LIST", {"default": None, "description": "可选：连接时使用原始路径，不连接时使用默认文件名", "tooltip": "可选的相对路径列表输入。如果连接此端口，将优先使用提供的路径列表保存图像，保持原始目录结构；不连接时将使用默认的文件名生成规则。"}),
                "prompt": ("PROMPT", {"default": None, "tooltip": "工作流提示词信息，用于保存到图像元数据中。"}),
                "extra_pnginfo": ("EXTRA_PNGINFO", {"default": None, "tooltip": "额外的PNG信息，包含完整工作流数据。"}),
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

    def save_images(self, 图像批量, 输出目录, 覆盖已存在文件, 保存格式, JPG_WEBP_压缩质量, 保存元数据, 相对路径列表=None, prompt=None, extra_pnginfo=None):
        """
        批量保存图像到指定目录，保持原始相对路径结构
        
        Args:
            图像批量 (torch.Tensor): 要保存的图像张量，形状为(B, H, W, C)或(H, W, C)
            输出目录 (str): 保存目录路径，支持相对路径和绝对路径
            覆盖已存在文件 (bool): 是否覆盖已存在的同名文件
            保存格式 (str): 保存格式，可选值："原格式"、"jpg"、"png"、"webp"
            JPG_WEBP_压缩质量 (int): JPG和WEBP格式的压缩质量（1-100）
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
        output_dir = Path(输出目录) if 输出目录.strip() else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果开启保存元数据，强制使用PNG格式
        if 保存元数据:
            print("保存元数据已开启，自动使用PNG格式")
            保存格式 = "png"
        
        # 如果没有提供相对路径列表，生成默认文件名
        if 相对路径列表 is None or len(相对路径列表) == 0:
            print("未连接相对路径列表，使用默认文件名")
            # 确定文件扩展名
            if 保存格式 != "原格式":
                ext = f".{保存格式}"
            else:
                ext = ".png"
            
            # 生成默认相对路径列表
            # 查找已存在的文件，确定起始序号
            existing_files = list(output_dir.glob(f"image_*{ext}"))
            if existing_files:
                # 提取现有文件的序号
                import re
                numbers = []
                for f in existing_files:
                    match = re.search(r'image_(\d+)', f.stem)
                    if match:
                        numbers.append(int(match.group(1)))
                start_num = max(numbers) + 1 if numbers else 1
            else:
                start_num = 1
            
            相对路径列表 = []
            for idx in range(len(图像批量)):
                filename = f"image_{start_num + idx:04d}{ext}"
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
                    save_kwargs["quality"] = JPG_WEBP_压缩质量
                
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
                "保存路径": ("STRING", {"default": "output", "placeholder": "保存目录路径，默认为output文件夹", "tooltip": "图像文件的保存目录路径。支持相对路径（基于ComfyUI根目录）和绝对路径。如果目录不存在，将自动创建。默认为output文件夹。"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否", "tooltip": "当目标位置已存在同名文件时是否覆盖。开启后将直接覆盖已存在的文件；关闭则自动生成不重复的文件名（添加数字后缀）避免覆盖。"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式", "tooltip": "图像保存格式。原格式：保持原始图像格式；jpg/webp：标准有损压缩格式，适合照片；png：无损压缩格式，支持透明度，适合图形和截图。"}),
                "压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效", "tooltip": "JPG和WEBP格式的压缩质量（1-100）。数值越高，图像质量越好，文件越大。推荐设置：照片用85-95，网页用70-85，测试用50-70。PNG格式此设置无效。"}),
                "保存元数据": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后保存图片元数据信息（如生成参数、工作流信息等），并自动使用PNG格式保存。关闭则不保存元数据，可使用任意格式。"}),
            },
            "optional": {
                "filename_text": ("STRING", {"default": "", "description": "从加载图像节点连接的文件名文本，不连接则使用自动生成的文件名", "tooltip": "从图像加载节点连接的文件名文本。如果连接此端口，将使用提供的前缀名称生成文件；不连接则使用自动生成的默认文件名（如frame_0001.png）。"}),
                "prompt": ("PROMPT", {"default": None, "tooltip": "工作流提示词信息，用于保存到图像元数据中。"}),
                "extra_pnginfo": ("EXTRA_PNGINFO", {"default": None, "tooltip": "额外的PNG信息，包含完整工作流数据。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("预览图像", "保存路径", "保存数量")
    FUNCTION = "save_and_preview"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "保存图像到指定路径并提供预览功能，支持批量保存视频帧序列和单张图像"

    def save_and_preview(self, 图像, 保存路径, 覆盖已存在文件, 保存格式, 压缩质量, 保存元数据, filename_text="", prompt=None, extra_pnginfo=None):
        """
        保存图像到指定路径并提供预览功能，支持批量保存视频帧序列和单张图像
        
        Args:
            图像 (torch.Tensor): 要保存的图像张量，形状为(B, H, W, C)或(H, W, C)
            保存路径 (str): 保存目录路径，支持相对路径和绝对路径
            覆盖已存在文件 (bool): 是否覆盖已存在的同名文件
            保存格式 (str): 保存格式，可选值："原格式"、"jpg"、"png"、"webp"
            压缩质量 (int): JPG和WEBP格式的压缩质量（1-100）
            保存元数据 (bool): 是否保存元数据信息
            filename_text (str, optional): 从图像加载节点连接的文件名文本
            prompt (dict, optional): 工作流提示词信息
            extra_pnginfo (dict, optional): 额外的PNG信息，包含完整工作流数据
        
        Returns:
            tuple: 包含以下元素的元组
                - 预览图像 (torch.Tensor): 与输入相同的图像张量，用于预览
                - 保存路径 (str): 实际保存路径
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
            return (图像, "", 0)
        
        # 处理保存路径
        save_dir = Path(保存路径) if 保存路径.strip() else Path("output")
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
            if filename_text:
                import re
                def clean_filename(name):
                    name = name.replace('/', '_').replace('\\', '_')
                    name = re.sub(r'[^\w\.\-]', '_', name)
                    return os.path.splitext(name)[0]  # 移除扩展名
                base = clean_filename(filename_text)
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
            return (图像[0:1], str(save_dir), saved_count)
        
        else:
            # 单张图像保存模式
            # 处理文件名和扩展名
            if not filename_text:
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
                
                filename = clean_filename(filename_text)
                
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
            return (图像, str(output_path), 1)
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return (图像, "", 0)


# --------------------------------------------------------------------------
# 条件图像保存器节点（根据布尔值分类保存）
# --------------------------------------------------------------------------
class ConditionalImageSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "要条件性保存的图像数据。根据布尔条件判断，将图像分类输出到不同的端口（True输出无水印图像，False输出有水印图像）。"}),
                "条件": ("BOOLEAN", {"default": False, "description": "True=无水印，False=有水印", "tooltip": "布尔条件判断。True（真）：输出无水印图像；False（假）：输出有水印图像和遮罩（仅当遮罩非空时）。用于图像分类和处理流程控制。"}),
                "启用分类保存": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "description": "关闭时不保存文件，但输出端口仍然有效", "tooltip": "是否启用分类保存功能。开启时将根据条件保存图像到相应子目录；关闭时仅输出数据到端口，不保存文件。便于调试和测试流程。"}),
                "基础保存路径": ("STRING", {"default": "output", "placeholder": "基础保存目录", "tooltip": "图像保存的基础目录路径。图像和遮罩将保存在此目录下的相应子目录中。支持相对路径（基于ComfyUI根目录）和绝对路径。"}),
                "True时子目录": ("STRING", {"default": "无水印", "placeholder": "留空则跳过保存", "tooltip": "当条件为True时图像保存的子目录名称。如果留空或仅包含空格，将跳过图像保存。目录不存在时会自动创建。"}),
                "False时子目录": ("STRING", {"default": "有水印", "placeholder": "留空则跳过保存", "tooltip": "当条件为False时图像保存的子目录名称。如果留空或仅包含空格，将跳过图像保存。目录不存在时会自动创建。"}),
                "False时遮罩子目录": ("STRING", {"default": "有水印遮罩", "placeholder": "留空则跳过保存", "tooltip": "当条件为False时遮罩保存的子目录名称。仅在False条件和遮罩非空时有效。如果留空或仅包含空格，将跳过遮罩保存。"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "tooltip": "当目标位置已存在同名文件时是否覆盖。开启后将直接覆盖已存在的文件；关闭则自动生成不重复的文件名（添加数字后缀）避免覆盖。"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式", "tooltip": "图像保存格式。原格式：保持原始图像格式；jpg/webp：标准有损压缩格式，适合照片；png：无损压缩格式，支持透明度，适合图形和截图。"}),
                "压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效", "tooltip": "JPG和WEBP格式的压缩质量（1-100）。数值越高，图像质量越好，文件越大。推荐设置：照片用85-95，网页用70-85，测试用50-70。PNG格式此设置无效。"}),
            },
            "optional": {
                "filename_text": ("STRING", {"default": "", "description": "从加载图像节点连接的文件名文本", "tooltip": "从图像加载节点连接的文件名文本。如果连接此端口，将使用提供的前缀名称生成文件；不连接则使用自动生成的默认文件名。"}),
                "遮罩": ("MASK", {"default": None, "description": "可选的遮罩输入", "tooltip": "可选的遮罩输入。仅在条件为False且遮罩非空时，遮罩才会被输出和保存。常用于需要保存图像掩码信息的后续处理。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("True图像", "False图像", "False遮罩")
    FUNCTION = "conditional_save"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "根据布尔条件将图像保存到不同的子目录并输出。True输出无水印图像，False输出有水印图像和遮罩（仅当遮罩非空时）。可通过开关控制是否保存文件"

    def conditional_save(self, 图像, 条件, 启用分类保存, 基础保存路径, True时子目录, False时子目录, False时遮罩子目录, 覆盖已存在文件, 保存格式, 压缩质量, filename_text="", 遮罩=None):
        # 准备输出（无论是否保存都需要）
        if 图像.ndim == 3:
            output_image = 图像.unsqueeze(0)
        else:
            output_image = 图像
        
        # 检查遮罩是否为空
        mask_is_empty = True
        output_mask = None
        if 遮罩 is not None:
            mask_sum = torch.sum(遮罩).item()
            if mask_sum > 0:
                mask_is_empty = False
                output_mask = 遮罩
        
        # 如果关闭分类保存，直接返回输出，不保存文件
        if not 启用分类保存:
            print(f"[分类保存已关闭] 跳过文件保存，仅输出数据")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            
            if 条件:
                # True: 无水印图像
                return (output_image, empty_image, empty_mask)
            else:
                # False: 有水印图像
                if not mask_is_empty and output_mask is not None:
                    return (empty_image, output_image, output_mask)
                else:
                    return (empty_image, output_image, empty_mask)
        
        # 启用分类保存时，执行保存逻辑
        # 验证图像是否有效
        is_valid, reason = is_valid_image_to_save(图像)
        if not is_valid:
            print(f"跳过保存: {reason}")
            # 返回空输出
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            if 条件:
                return (empty_image, empty_image, empty_mask)
            else:
                return (empty_image, empty_image, empty_mask)
        
        # 根据条件选择子目录
        if 条件:
            sub_dir = True时子目录
            分类结果 = "无水印"
        else:
            sub_dir = False时子目录
            分类结果 = "有水印"
        
        # 检查是否跳过图像保存（子目录名为空时跳过）
        should_skip_image_save = not sub_dir or sub_dir.strip() == ""
        
        # 检查是否需要保存遮罩
        need_save_mask = False
        if not 条件 and 遮罩 is not None and False时遮罩子目录 and False时遮罩子目录.strip() != "":
            try:
                mask_sum = torch.sum(遮罩).item()
                if mask_sum > 0:
                    need_save_mask = True
            except:
                pass
        
        # 对于False（有水印）的情况，即使跳过图像保存，也要检查是否需要保存遮罩
        if should_skip_image_save and not need_save_mask:
            print(f"[{分类结果}] 子目录名为空，跳过保存")
            # 准备输出并返回
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            
            if 条件:
                return (output_image, empty_image, empty_mask)
            else:
                if not mask_is_empty and output_mask is not None:
                    return (empty_image, output_image, output_mask)
                else:
                    return (empty_image, output_image, empty_mask)
        
        # 构建基础路径
        base_path = Path(基础保存路径) if 基础保存路径.strip() else Path("output")
        
        # 准备文件名（用于图像和遮罩）
        import re
        def clean_filename(name):
            name = name.replace('/', '_').replace('\\', '_')
            name = re.sub(r'[^\w\.\-]', '_', name)
            return name
        
        # 保存图像（如果不跳过）
        if not should_skip_image_save:
            save_dir = base_path / sub_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 处理文件名
            if not filename_text:
                # 生成默认序号文件名
                base = "image"
                if 保存格式 != "原格式":
                    ext = f".{保存格式}"
                else:
                    ext = ".png"
                counter = 1
                while True:
                    filename = f"{base}_{counter:04d}{ext}"
                    output_path = save_dir / filename
                    if not output_path.exists():
                        break
                    counter += 1
                    if counter > 99999:
                        raise Exception("超过最大尝试次数（99999），无法生成唯一文件名")
            else:
                filename = clean_filename(filename_text)
                
                # 处理保存格式
                if 保存格式 != "原格式":
                    filename = os.path.splitext(filename)[0] + f".{保存格式}"
                
                output_path = save_dir / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查文件是否存在
            if output_path.exists() and not 覆盖已存在文件:
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
                        save_format = 'PNG'
                else:
                    if 保存格式 in ["jpg", "webp"]:
                        save_kwargs["quality"] = 压缩质量
                    save_format = 'WebP' if 保存格式 == 'webp' else (保存格式.upper() if 保存格式 != 'jpg' else 'JPEG')
                
                if not output_path.suffix:
                    output_path = output_path.with_suffix(f".{save_format.lower()}")
                
                img.save(output_path, format=save_format, **save_kwargs)
                print(f"[{分类结果}] 图像已保存: {output_path}")
            except Exception as e:
                print(f"[{分类结果}] 保存图像失败: {str(e)}")
                output_path = None
        else:
            print(f"[{分类结果}] 子目录名为空，跳过图像保存")
            output_path = None
        
        # 保存遮罩（独立于图像保存）
        if not 条件 and 遮罩 is not None:
            mask_sum = torch.sum(遮罩).item()
            if mask_sum > 0:
                # 检查遮罩子目录是否需要跳过
                should_skip_mask = not False时遮罩子目录 or False时遮罩子目录.strip() == ""
                
                if should_skip_mask:
                    print(f"[{分类结果}] 遮罩子目录名为空，跳过遮罩保存")
                else:
                    # 构建遮罩保存路径
                    mask_save_dir = base_path / False时遮罩子目录
                    mask_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 确定遮罩文件名
                    if output_path is not None:
                        # 如果图像已保存，使用相同的文件名
                        mask_filename = os.path.splitext(output_path.name)[0] + '.png'
                    elif filename_text:
                        # 如果有filename_text，使用它
                        mask_filename = os.path.splitext(clean_filename(filename_text))[0] + '.png'
                    else:
                        # 生成默认文件名
                        counter = 1
                        while True:
                            mask_filename = f"mask_{counter:04d}.png"
                            if not (mask_save_dir / mask_filename).exists():
                                break
                            counter += 1
                            if counter > 99999:
                                print(f"[{分类结果}] 遮罩保存失败: 无法生成唯一文件名（超过99999次尝试）")
                                mask_filename = None
                                break
                    
                    if mask_filename:
                        mask_output_path = mask_save_dir / mask_filename
                        
                        # 检查遮罩文件是否存在
                        if mask_output_path.exists() and not 覆盖已存在文件:
                            base = mask_output_path.stem
                            ext = mask_output_path.suffix
                            counter = 1
                            while True:
                                new_filename = f"{base}_{counter:04d}{ext}"
                                new_mask_path = mask_output_path.parent / new_filename
                                if not new_mask_path.exists():
                                    mask_output_path = new_mask_path
                                    break
                                counter += 1
                                if counter > 99999:
                                    print(f"[{分类结果}] 遮罩保存失败: 无法生成唯一文件名（超过99999次尝试）")
                                    break
                        
                        try:
                            # 处理遮罩张量
                            if 遮罩.ndim == 3 and 遮罩.shape[0] == 1:
                                mask_tensor = 遮罩[0]
                            elif 遮罩.ndim == 2:
                                mask_tensor = 遮罩
                            else:
                                mask_tensor = 遮罩[0] if 遮罩.ndim == 3 else 遮罩
                            
                            mask_np = mask_tensor.cpu().numpy()
                            mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
                            mask_img = Image.fromarray(mask_np, mode='L')
                            mask_img.save(mask_output_path, format='PNG')
                            print(f"[{分类结果}] 遮罩已保存: {mask_output_path}")
                        except Exception as mask_e:
                            print(f"[{分类结果}] 保存遮罩失败: {str(mask_e)}")
        
        # 准备输出
        # 确保图像是4维张量 (batch, height, width, channels)
        if 图像.ndim == 3:
            output_image = 图像.unsqueeze(0)
        else:
            output_image = 图像
        
        # 检查遮罩是否为空
        mask_is_empty = True
        output_mask = None
        if 遮罩 is not None:
            # 检查遮罩是否全为0（空遮罩）
            mask_sum = torch.sum(遮罩).item()
            if mask_sum > 0:
                mask_is_empty = False
                output_mask = 遮罩
        
        # 根据条件返回不同的输出
        if 条件:
            # True: 无水印图像
            # 返回: True图像, 空的False图像, 空的遮罩
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (output_image, empty_image, empty_mask)
        else:
            # False: 有水印图像
            # 只有当遮罩非空时才输出遮罩，否则输出空遮罩
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            if not mask_is_empty and output_mask is not None:
                print(f"[{分类结果}] 检测到有效遮罩，输出图像和遮罩用于重绘")
                return (empty_image, output_image, output_mask)
            else:
                print(f"[{分类结果}] 遮罩为空，仅输出图像")
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
                return (empty_image, output_image, empty_mask)


# --------------------------------------------------------------------------
# 条件图像目录保存器节点（批量保存，保持目录结构）
# --------------------------------------------------------------------------
class ConditionalImageDirectorySaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像批量": ("IMAGE", {"tooltip": "要条件性批量保存的图像数据。根据布尔条件判断，将图像分类输出到不同的端口（True输出无水印图像，False输出有水印图像和遮罩），并保存到相应子目录。"}),
                "条件": ("BOOLEAN", {"default": False, "description": "True=无水印，False=有水印", "tooltip": "布尔条件判断。True（真）：输出无水印图像；False（假）：输出有水印图像和遮罩（仅当遮罩非空时）。用于图像批量分类和处理流程控制。"}),
                "启用分类保存": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "description": "关闭时不保存文件，但输出端口仍然有效", "tooltip": "是否启用分类保存功能。开启时将根据条件保存图像到相应子目录；关闭时仅输出数据到端口，不保存文件。便于调试和测试流程。"}),
                "基础输出目录": ("STRING", {"default": "output", "multiline": False, "placeholder": "基础保存目录", "tooltip": "图像保存的基础目录路径。图像和遮罩将保存在此目录下的相应子目录中。支持相对路径（基于ComfyUI根目录）和绝对路径。"}),
                "True时子目录": ("STRING", {"default": "无水印", "placeholder": "留空则跳过保存", "tooltip": "当条件为True时图像保存的子目录名称。如果留空或仅包含空格，将跳过图像保存。目录不存在时会自动创建。"}),
                "False时子目录": ("STRING", {"default": "有水印", "placeholder": "留空则跳过保存", "tooltip": "当条件为False时图像保存的子目录名称。如果留空或仅包含空格，将跳过图像保存。目录不存在时会自动创建。"}),
                "False时遮罩子目录": ("STRING", {"default": "有水印遮罩", "placeholder": "留空则跳过保存", "tooltip": "当条件为False时遮罩保存的子目录名称。仅在False条件和遮罩非空时有效。如果留空或仅包含空格，将跳过遮罩保存。"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "tooltip": "当目标位置已存在同名文件时是否覆盖。开启后将直接覆盖已存在的文件；关闭则自动生成不重复的文件名（添加数字后缀）避免覆盖。"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式", "tooltip": "图像保存格式。原格式：保持原始图像格式；jpg/webp：标准有损压缩格式，适合照片；png：无损压缩格式，支持透明度，适合图形和截图。"}),
                "JPG_WEBP_压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效", "tooltip": "JPG和WEBP格式的压缩质量（1-100）。数值越高，图像质量越好，文件越大。推荐设置：照片用85-95，网页用70-85，测试用50-70。PNG格式此设置无效。"}),
            },
            "optional": {
                "相对路径列表": ("LIST", {"default": None, "description": "可选：连接时使用原始路径，不连接时使用默认文件名", "tooltip": "相对路径列表。如果连接此端口，将使用原始相对路径保存文件（保持目录结构）；不连接则使用自动生成的默认文件名（image_0001.png等）。"}),
                "遮罩": ("MASK", {"default": None, "description": "可选的遮罩输入", "tooltip": "可选的遮罩输入。仅在条件为False且遮罩非空时，遮罩才会被输出和保存到对应的遮罩子目录。常用于需要保存图像掩码信息的批量处理。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("True图像", "False图像", "False遮罩")
    FUNCTION = "conditional_save_batch"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "根据布尔条件批量保存图像到不同的子目录，保持原始相对路径结构。True输出无水印图像，False输出有水印图像和遮罩（仅当遮罩非空时）。可通过开关控制是否保存文件"

    def conditional_save_batch(self, 图像批量, 条件, 启用分类保存, 基础输出目录, True时子目录, False时子目录, False时遮罩子目录, 覆盖已存在文件, 保存格式, JPG_WEBP_压缩质量, 相对路径列表=None, 遮罩=None):
        # 准备输出（无论是否保存都需要）
        if 图像批量.ndim == 3:
            output_images = 图像批量.unsqueeze(0)
        else:
            output_images = 图像批量
        
        # 检查遮罩是否为空
        mask_is_empty = True
        output_mask = None
        if 遮罩 is not None:
            mask_sum = torch.sum(遮罩).item()
            if mask_sum > 0:
                mask_is_empty = False
                output_mask = 遮罩
        
        # 如果关闭分类保存，直接返回输出，不保存文件
        if not 启用分类保存:
            print(f"[分类保存已关闭] 跳过文件保存，仅输出数据")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            
            if 条件:
                # True: 无水印图像
                return (output_images, empty_image, empty_mask)
            else:
                # False: 有水印图像
                if not mask_is_empty and output_mask is not None:
                    return (empty_image, output_images, output_mask)
                else:
                    return (empty_image, output_images, empty_mask)
        
        # 启用分类保存时，执行保存逻辑
        # 根据条件选择子目录
        if 条件:
            sub_dir = True时子目录
            分类结果 = "无水印"
        else:
            sub_dir = False时子目录
            分类结果 = "有水印"
        
        # 检查是否跳过图像保存（子目录名为空时跳过）
        should_skip_image_save = not sub_dir or sub_dir.strip() == ""
        
        # 检查是否需要保存遮罩
        need_save_mask = False
        if not 条件 and 遮罩 is not None and False时遮罩子目录 and False时遮罩子目录.strip() != "":
            try:
                mask_sum = torch.sum(遮罩).item()
                if mask_sum > 0:
                    need_save_mask = True
            except:
                pass
        
        # 对于False（有水印）的情况，即使跳过图像保存，也要检查是否需要保存遮罩
        if should_skip_image_save and not need_save_mask:
            print(f"[{分类结果}] 子目录名为空，跳过保存")
            # 准备输出并返回
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            
            if 条件:
                return (output_images, empty_image, empty_mask)
            else:
                if not mask_is_empty and output_mask is not None:
                    return (empty_image, output_images, output_mask)
                else:
                    return (empty_image, output_images, empty_mask)
        
        # 构建基础路径
        base_path = Path(基础输出目录) if 基础输出目录.strip() else Path("output")
        
        # 只有在不跳过图像保存时才创建图像保存目录
        if not should_skip_image_save:
            output_dir = base_path / sub_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None
            print(f"[{分类结果}] 子目录名为空，跳过图像保存")
        
        # 如果没有提供相对路径列表，生成默认文件名
        if (相对路径列表 is None or len(相对路径列表) == 0) and not should_skip_image_save:
            print(f"[{分类结果}] 未连接相对路径列表，使用默认文件名")
            # 确定文件扩展名
            if 保存格式 != "原格式":
                ext = f".{保存格式}"
            else:
                ext = ".png"
            
            # 生成默认相对路径列表
            # 查找已存在的文件，确定起始序号
            existing_files = list(output_dir.glob(f"image_*{ext}"))
            if existing_files:
                # 提取现有文件的序号
                import re
                numbers = []
                for f in existing_files:
                    match = re.search(r'image_(\d+)', f.stem)
                    if match:
                        numbers.append(int(match.group(1)))
                start_num = max(numbers) + 1 if numbers else 1
            else:
                start_num = 1
            
            相对路径列表 = []
            for idx in range(len(图像批量)):
                filename = f"image_{start_num + idx:04d}{ext}"
                相对路径列表.append(filename)
        
        if len(图像批量) != len(相对路径列表):
            print(f"[{分类结果}] 错误: 图像数量 ({len(图像批量)}) 与路径数量 ({len(相对路径列表)}) 不匹配")
            return ()

        saved_count = 0
        skipped_count = 0
        mask_saved_count = 0
        
        # 如果是False（有水印）且遮罩非空，准备遮罩保存目录
        mask_save_dir = None
        if not 条件 and 遮罩 is not None:
            mask_sum = torch.sum(遮罩).item()
            if mask_sum > 0:
                # 检查遮罩子目录是否需要跳过
                should_skip_mask = not False时遮罩子目录 or False时遮罩子目录.strip() == ""
                
                if should_skip_mask:
                    print(f"[{分类结果}] 遮罩子目录名为空，跳过遮罩保存")
                else:
                    mask_save_dir = base_path / False时遮罩子目录
                    mask_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果需要保存图像，执行批量保存
        if not should_skip_image_save:
            for idx, (img_tensor, rel_path) in enumerate(zip(图像批量, 相对路径列表)):
                # 验证图像是否有效
                is_valid, reason = is_valid_image_to_save(img_tensor)
                if not is_valid:
                    print(f"[{分类结果}] 跳过保存 {rel_path}: {reason}")
                    skipped_count += 1
                    continue
                
                output_path = output_dir / rel_path
                # 如果是原格式则保留原始扩展名，否则使用指定格式
                if 保存格式 != '原格式':
                    output_path = output_path.with_suffix(f'.{保存格式}')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and not 覆盖已存在文件:
                    print(f"[{分类结果}] 跳过已存在文件: {output_path}")
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
                        save_kwargs["quality"] = JPG_WEBP_压缩质量
                    img.save(output_path, **save_kwargs)
                    saved_count += 1
                    print(f"[{分类结果}] 已保存图像: {output_path}")
                except Exception as e:
                    print(f"[{分类结果}] 保存图像失败 {rel_path}: {str(e)}")
        
        # 保存遮罩（独立于图像保存）
        if mask_save_dir is not None and 遮罩 is not None:
            # 如果没有相对路径列表，生成默认文件名
            if 相对路径列表 is None or len(相对路径列表) == 0:
                相对路径列表 = []
                for idx in range(len(图像批量)):
                    filename = f"mask_{idx:04d}.png"
                    相对路径列表.append(filename)
            
            for idx, rel_path in enumerate(相对路径列表):
                if idx >= len(图像批量):
                    break
                
                try:
                    # 获取对应索引的遮罩
                    if 遮罩.ndim == 3 and idx < 遮罩.shape[0]:
                        mask_tensor = 遮罩[idx]
                    elif 遮罩.ndim == 2:
                        mask_tensor = 遮罩
                    else:
                        mask_tensor = None
                    
                    if mask_tensor is not None:
                        # 构建遮罩保存路径（保持相对路径结构）
                        mask_rel_path = os.path.splitext(rel_path)[0] + '.png'
                        mask_output_path = mask_save_dir / mask_rel_path
                        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 检查遮罩文件是否存在
                        if mask_output_path.exists() and not 覆盖已存在文件:
                            base = mask_output_path.stem
                            ext = mask_output_path.suffix
                            counter = 1
                            while True:
                                new_filename = f"{base}_{counter:04d}{ext}"
                                new_mask_path = mask_output_path.parent / new_filename
                                if not new_mask_path.exists():
                                    mask_output_path = new_mask_path
                                    break
                                counter += 1
                                if counter > 99999:
                                    print(f"[{分类结果}] 遮罩保存失败 {rel_path}: 无法生成唯一文件名（超过99999次尝试）")
                                    break
                        
                        mask_np = mask_tensor.cpu().numpy()
                        mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
                        mask_img = Image.fromarray(mask_np, mode='L')
                        mask_img.save(mask_output_path, format='PNG')
                        mask_saved_count += 1
                        print(f"[{分类结果}] 已保存遮罩: {mask_output_path}")
                except Exception as mask_e:
                    print(f"[{分类结果}] 保存遮罩失败 {rel_path}: {str(mask_e)}")
        
        if mask_saved_count > 0:
            print(f"[{分类结果}] 批量保存完成: {saved_count} 张图像已保存, {mask_saved_count} 张遮罩已保存, {skipped_count} 张已跳过")
            print(f"  - 图像目录: {output_dir}")
            print(f"  - 遮罩目录: {mask_save_dir}")
        else:
            print(f"[{分类结果}] 批量保存完成: {saved_count} 张已保存, {skipped_count} 张已跳过, 目标目录: {output_dir}")
        
        # 根据条件返回不同的输出（output_images, output_mask 已在方法开头准备好）
        if 条件:
            # True: 无水印图像
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (output_images, empty_image, empty_mask)
        else:
            # False: 有水印图像
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            if not mask_is_empty and output_mask is not None:
                print(f"[{分类结果}] 检测到有效遮罩，输出图像和遮罩用于重绘")
                return (empty_image, output_images, output_mask)
            else:
                print(f"[{分类结果}] 遮罩为空，仅输出图像")
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
                return (empty_image, output_images, empty_mask)


# --------------------------------------------------------------------------
# 节点注册
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "zyf_ImageDirectoryLoader": ImageDirectoryLoader,
    "zyf_ImageDirectorySaver": ImageDirectorySaver,
    "zyf_ImageSaveWithPreview": ImageSaveWithPreview,
    "zyf_ConditionalImageSaver": ConditionalImageSaver,
    "zyf_ConditionalImageDirectorySaver": ConditionalImageDirectorySaver,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zyf_ImageDirectoryLoader": "图像目录加载器",
    "zyf_ImageDirectorySaver": "图像目录保存器",
    "zyf_ImageSaveWithPreview": "图像保存与预览器",
    "zyf_ConditionalImageSaver": "条件图像保存器",
    "zyf_ConditionalImageDirectorySaver": "条件图像目录保存器",

}

NODE_DESCRIPTION_MAPPINGS = {
    "zyf_ImageDirectoryLoader": "从指定目录加载图像，支持多种排序和过滤选项",
    "zyf_ImageDirectorySaver": "将图像批量保存到指定目录，保持原始相对路径结构",
    "zyf_ImageSaveWithPreview": "保存单张图像到指定路径并提供预览，支持连接加载节点的文件名",
    "zyf_ConditionalImageSaver": "根据布尔条件将图像分类保存到不同目录（无水印/有水印）",
    "zyf_ConditionalImageDirectorySaver": "根据布尔条件批量保存图像到不同目录，保持原始相对路径结构",

}