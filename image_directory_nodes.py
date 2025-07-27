import os
import re
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
import hashlib

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
# 图像目录加载器节点
# --------------------------------------------------------------------------
class ImageDirectoryLoader:
    _memory = {}  # 记忆库: {(目录路径, 任务批次编号): 最后加载位置}
    _auto_index = {}  # 自动索引库: {(目录路径, 任务批次编号): 当前索引}
    
    @classmethod
    def _get_cache_file(cls):
        """获取缓存文件路径"""
        cache_dir = Path(__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "auto_index.json"
    
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
        # 使用路径和批次号的组合作为键，确保跨会话一致性
        key_str = f"{目录路径}#{任务批次编号}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "目录路径": ("STRING", {"default": "", "multiline": False, "placeholder": "请输入要加载的图像目录路径"}),
                "起始索引": ("INT", {"default": 0, "min": 0, "step": 1, "description": "从第几张图片开始"}),
                "单张顺序加载": ("BOOLEAN", {"default": True, "description": "启用后按顺序每次仅加载一张图片，自动递增索引实现连续加载"}),
                "启用记忆功能": ("BOOLEAN", {"default": False, "description": "启用后记住最后加载位置，支持断点续传"}),
                "任务批次编号": ("STRING", {"default": "batch01", "description": "任务批次标识，变化时重置记忆和自动索引"}),
                "智能队列建议": ("BOOLEAN", {"default": True, "description": "启用后会在控制台显示剩余图片数量和建议的队列次数"}),
                "sort_method": (["按名称", "按数字", "按修改时间"], {"default": "按名称", "description": "图片排序方式"}),
                "递归搜索子目录": ("BOOLEAN", {"default": True, "description": "是否递归查找所有子文件夹"}),
                "文件扩展名过滤": ("STRING", {"default": "", "placeholder": "用逗号分隔，如: jpg,png", "description": "留空则加载所有支持的图片格式(jpg,jpeg,png,bmp,webp,tiff)"}),
                "加载失败跳过": ("BOOLEAN", {"default": True, "description": "加载失败时是否跳过"}),
                "转换为RGBA": ("BOOLEAN", {"default": False, "description": "是否将图像转换为RGBA透明通道格式，启用后将以PNG格式保存"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "STRING", "INT", "INT")
    RETURN_NAMES = ("图像", "相对路径", "filename_text", "可用总数", "剩余未处理")
    FUNCTION = "load_images"
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "从指定目录批量加载图片，支持递归、排序、扩展名过滤等功能。单张顺序加载模式使用持久化缓存，确保每次运行自动加载下一张图片。"

    def load_images(self, 目录路径, 起始索引, 单张顺序加载, 启用记忆功能, 任务批次编号, 智能队列建议, sort_method, 递归搜索子目录, 文件扩展名过滤, 加载失败跳过, 转换为RGBA):
        if not os.path.isdir(目录路径):
            print(f"错误: 目录 '{目录路径}' 不存在")
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", 0)

        # 支持自定义扩展名，留空则加载所有常见图片格式
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
        total_available = len(image_paths)
        if total_available == 0:
            print("未找到任何图像文件")
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", 0)

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

        # 确定加载数量
        if 单张顺序加载:
            最大加载数量 = 1
        else:
            # 批量模式下加载所有剩余图片
            最大加载数量 = len(image_paths)
            
        # 确定起始索引（记忆功能处理）
        if 单张顺序加载:
            # 加载自动索引缓存
            ImageDirectoryLoader._load_auto_index()
            
            # 生成缓存键
            cache_key = ImageDirectoryLoader._get_key(目录路径, 任务批次编号)
            
            # 检查是否需要重置（路径或批次号变化）
            last_config_key = "_last_config"
            current_config = f"{目录路径}#{任务批次编号}"
            
            if (last_config_key not in ImageDirectoryLoader._auto_index or 
                ImageDirectoryLoader._auto_index[last_config_key] != current_config):
                # 路径或批次号发生变化，清理相关缓存
                if 启用记忆功能:
                    ImageDirectoryLoader._memory.clear()
                    print(f"检测到路径或批次号变化，已清理记忆缓存")
                
                # 重置当前批次的自动索引
                ImageDirectoryLoader._auto_index[cache_key] = 起始索引
                ImageDirectoryLoader._auto_index[last_config_key] = current_config
                ImageDirectoryLoader._save_auto_index()
                print(f"检测到配置变化，已重置自动索引到起始位置: {起始索引}")
            
            # 单张顺序加载模式：自动递增索引
            if cache_key not in ImageDirectoryLoader._auto_index:
                ImageDirectoryLoader._auto_index[cache_key] = 起始索引
                ImageDirectoryLoader._save_auto_index()
            
            start = ImageDirectoryLoader._auto_index[cache_key]
        else:
            key = (目录路径, 任务批次编号)
            if 启用记忆功能:
                if key in ImageDirectoryLoader._memory:
                    start = ImageDirectoryLoader._memory[key]
                else:
                    start = 起始索引
            else:
                start = 起始索引
        
        end = start + 最大加载数量
        selected_paths = image_paths[start:end]
        total_loaded = len(selected_paths)
        if total_loaded == 0:
            print(f"未选择任何图像。起始索引 {start} 可能过高")
            # 计算剩余未处理数量
            remaining = max(0, total_available - start) if 启用记忆功能 or 单张顺序加载 else 0
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
            remaining = max(0, total_available - start) if 启用记忆功能 or 单张顺序加载 else 0
            return (torch.zeros((0, 1, 1, 3), dtype=torch.float32), [], "", total_available, remaining)

        # 单张顺序加载模式处理
        if 单张顺序加载 and len(images) > 0:
            current_index = start + 1
            has_next = current_index < total_available

            # 更新自动索引（循环到末尾时重置为0）
            cache_key = ImageDirectoryLoader._get_key(目录路径, 任务批次编号)
            if has_next:
                ImageDirectoryLoader._auto_index[cache_key] = current_index
            else:
                ImageDirectoryLoader._auto_index[cache_key] = 0  # 循环回到开始
                print(f"已加载完所有图像，下次将从第一张开始")
            
            # 保存自动索引到文件
            ImageDirectoryLoader._save_auto_index()

            # 更新记忆位置（如果启用记忆功能）
            if 启用记忆功能:
                key = (目录路径, 任务批次编号)
                ImageDirectoryLoader._memory[key] = current_index

            # 根据RGBA设置调整文件名
            if selected_paths and 转换为RGBA:
                original_name = selected_paths[0].name
                filename_text = os.path.splitext(original_name)[0] + '.png'
            else:
                filename_text = selected_paths[0].name if selected_paths else ""
            
            # 计算剩余未处理数量
            remaining = max(0, total_available - current_index)
            
            # 智能提示信息
            if 智能队列建议 and 启用记忆功能 and remaining > 0:
                print(f"当前加载: {filename_text} (索引: {start}/{total_available-1}) - 剩余未处理: {remaining}张")
                print(f"💡 智能建议: 队列设置为 {remaining} 次可完成剩余图片处理")
            elif 智能队列建议:
                print(f"当前加载: {filename_text} (索引: {start}/{total_available-1}) - 剩余: {remaining}张")
            else:
                print(f"当前加载: {filename_text} (索引: {start}/{total_available-1})")
            
            return (images[0], [relative_paths[0]], filename_text, total_available, remaining)
        
        batch_images = torch.cat(images, dim=0)
        # 批量加载模式下，剩余未处理数量为0（因为一次性加载了指定数量）
        remaining = 0
        return (batch_images, relative_paths, "", total_available, remaining)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 当启用单张顺序加载或记忆功能时，强制重新执行以避免缓存
        if kwargs.get('单张顺序加载', False) or kwargs.get('启用记忆功能', False):
            return float("NaN")
        # 对于普通批量加载，使用目录内容的哈希值来判断是否需要重新执行
        else:
            import hashlib
            目录路径 = kwargs.get('目录路径', '')
            if not os.path.isdir(目录路径):
                return float("NaN")
            
            # 获取目录中所有图像文件的修改时间和大小
            文件扩展名过滤 = kwargs.get('文件扩展名过滤', '')
            递归搜索子目录 = kwargs.get('递归搜索子目录', False)
            
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
            
            # 创建基于文件信息的哈希值
            hash_input = ""
            for img_path in sorted(image_paths):
                try:
                    stat = img_path.stat()
                    hash_input += f"{img_path}:{stat.st_mtime}:{stat.st_size};"
                except:
                    continue
            
            if hash_input:
                return hashlib.md5(hash_input.encode()).hexdigest()
            else:
                return float("NaN")

# --------------------------------------------------------------------------
# 图像目录保存器节点
# --------------------------------------------------------------------------
class ImageDirectorySaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像批量": ("IMAGE",),
                "相对路径列表": ("LIST", {"forceInput": True}),
                "输出目录": ("STRING", {"default": "ComfyUI/output", "multiline": False, "placeholder": "保存到哪个目录"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式"}),
                "JPG_WEBP_压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "批量保存图像到指定目录，保持原始相对路径结构"

    def save_images(self, 图像批量, 相对路径列表, 输出目录, 覆盖已存在文件, 保存格式, JPG_WEBP_压缩质量):
        output_dir = Path(输出目录)
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(图像批量) != len(相对路径列表):
            print(f"错误: 图像数量 ({len(图像批量)}) 与路径数量 ({len(相对路径列表)}) 不匹配")
            return ()

        for img_tensor, rel_path in zip(图像批量, 相对路径列表):
            output_path = output_dir / rel_path
            # 如果是原格式则保留原始扩展名，否则使用指定格式
            if 保存格式 != '原格式':
                output_path = output_path.with_suffix(f'.{保存格式}')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not 覆盖已存在文件:
                print(f"跳过已存在文件: {output_path}")
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
                print(f"已保存图像: {output_path}")
            except Exception as e:
                print(f"保存图像失败 {output_path}: {str(e)}")
        return ()

# --------------------------------------------------------------------------
# 图像保存与预览节点
# --------------------------------------------------------------------------
class ImageSaveWithPreview:
    @classmethod
    def INPUT_TYPES(s):
        comfyui_root = Path(__file__).parent.parent.parent.parent
        default_output_path = str(comfyui_root / "output")
        return {
            "required": {
                "图像": ("IMAGE",),
                "保存路径": ("STRING", {"default": "ComfyUI/output", "placeholder": "保存目录路径，默认为output文件夹"}),
                "覆盖已存在文件": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "保存格式": (["原格式", "jpg", "png", "webp"], {"default": "原格式"}),
                "压缩质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "description": "仅jpg/webp有效"}),
            },
            "optional": {
                "filename_text": ("STRING", {"default": "", "description": "从加载图像节点连接的文件名文本，不连接则使用自动生成的文件名"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("预览图像", "保存路径")
    FUNCTION = "save_and_preview"
    OUTPUT_NODE = True
    CATEGORY = "目录加载与保存"
    DESCRIPTION = "保存图像到指定路径并提供预览功能，支持连接加载图像节点的文件名"

    def save_and_preview(self, 图像, 保存路径, 覆盖已存在文件, 保存格式, 压缩质量, filename_text=""):
        # 处理保存路径
        if not 保存路径.strip():
            comfyui_root = Path(__file__).parent.parent.parent.parent
            save_dir = comfyui_root / "ComfyUI" / "output"
        else:
            save_dir = Path(保存路径)
        save_dir.mkdir(parents=True, exist_ok=True)

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
                filename = f"{base}_{counter:03d}{ext}"
                output_path = save_dir / filename
                if not output_path.exists():
                    break
                counter += 1
                if counter > 999:
                    raise Exception("超过最大尝试次数，无法生成唯一文件名")
        else:
                # 清理文件名，移除非法字符和路径分隔符
                import re
                def clean_filename(name):
                    # 替换路径分隔符为下划线
                    name = name.replace('/', '_').replace('\\', '_')
                    # 移除非字母数字、点、下划线、连字符的字符
                    name = re.sub(r'[^\w\.\-]', '_', name)
                    # 移除常见图像扩展名后的多余字符
                    image_exts = r'(jpg|jpeg|png|bmp|webp|tiff)'
                    # 匹配最后一个有效的图像扩展名并截断后面的内容
                    pattern = rf'\.({image_exts})(_.*)$'
                    name = re.sub(pattern, r'.\1', name, flags=re.IGNORECASE)
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
                new_filename = f"{base}_{counter:03d}{ext}"
                new_output_path = output_path.parent / new_filename
                if not new_output_path.exists():
                    output_path = new_output_path
                    break
                counter += 1
                if counter > 999:
                    raise Exception("超过最大尝试次数，无法生成唯一文件名")

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
            img.save(output_path, format=save_format, **save_kwargs)
            print(f"图像已保存: {output_path}")
            return (图像, str(output_path))
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return (图像, "")


# --------------------------------------------------------------------------
# 节点注册
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "zyf_ImageDirectoryLoader": ImageDirectoryLoader,
    "zyf_ImageDirectorySaver": ImageDirectorySaver,
    "zyf_ImageSaveWithPreview": ImageSaveWithPreview,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "zyf_ImageDirectoryLoader": "图像目录加载器",
    "zyf_ImageDirectorySaver": "图像目录保存器",
    "zyf_ImageSaveWithPreview": "图像保存与预览器",

}

NODE_DESCRIPTION_MAPPINGS = {
    "zyf_ImageDirectoryLoader": "从指定目录加载图像，支持多种排序和过滤选项",
    "zyf_ImageDirectorySaver": "将图像批量保存到指定目录，保持原始相对路径结构",
    "zyf_ImageSaveWithPreview": "保存单张图像到指定路径并提供预览，支持连接加载节点的文件名",

}