from .image_directory_nodes import NODE_CLASS_MAPPINGS as IMAGE_NODE_CLASS_MAPPINGS
from .image_directory_nodes import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_NODE_DISPLAY_NAME_MAPPINGS
from .video_directory_nodes import NODE_CLASS_MAPPINGS as VIDEO_NODE_CLASS_MAPPINGS
from .video_directory_nodes import NODE_DISPLAY_NAME_MAPPINGS as VIDEO_NODE_DISPLAY_NAME_MAPPINGS

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {**IMAGE_NODE_CLASS_MAPPINGS, **VIDEO_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**IMAGE_NODE_DISPLAY_NAME_MAPPINGS, **VIDEO_NODE_DISPLAY_NAME_MAPPINGS}

# Web前端扩展目录
WEB_DIRECTORY = "./js"

# 导入服务器扩展（注册API路由）
try:
    from . import server
except Exception as e:
    print(f"[zyf_image_directory_nodes] 服务器扩展加载失败: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
