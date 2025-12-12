# 📁 项目结构说明

本项目是一个ComfyUI自定义节点插件，用于批量处理图像和视频文件。以下是项目的文件结构和主要组件说明：

## 🏠 项目根目录

```
zyf_image_directory_nodes/
├── __init__.py                    # 节点注册文件，合并图像和视频节点
├── image_directory_nodes.py       # 图像处理核心节点实现
├── video_directory_nodes.py       # 视频处理核心节点实现
├── README.md                      # 项目说明文档
├── example_image_workflow.json    # 图像处理示例工作流
├── example_video_workflow.json    # 视频处理示例工作流
├── .cache/                        # 自动索引缓存目录
├── docs/                          # 文档目录
├── __pycache__/                   # Python缓存目录
└── 测试文件/                       # 各种测试脚本
```

## 📋 主要文件说明

### 核心实现文件

#### `__init__.py`
- **功能**：节点注册文件，将图像和视频节点合并并注册到ComfyUI
- **核心逻辑**：
  ```python
  from .image_directory_nodes import NODE_CLASS_MAPPINGS as IMAGE_NODE_CLASS_MAPPINGS
  from .video_directory_nodes import NODE_CLASS_MAPPINGS as VIDEO_NODE_CLASS_MAPPINGS
  NODE_CLASS_MAPPINGS = {**IMAGE_NODE_CLASS_MAPPINGS, **VIDEO_NODE_CLASS_MAPPINGS}
  NODE_DISPLAY_NAME_MAPPINGS = {**IMAGE_NODE_DISPLAY_NAME_MAPPINGS, **VIDEO_NODE_DISPLAY_NAME_MAPPINGS}
  ```

#### `image_directory_nodes.py`
- **功能**：图像处理核心节点实现
- **主要节点**：
  - `ImageDirectoryLoader` - 图像目录加载器
  - `ImageDirectorySaver` - 图像目录保存器
  - `ImageSaveWithPreview` - 图像保存与预览
  - `ConditionImageSaver` - 条件图像保存器
- **主要功能**：批量图像加载、保存、条件保存、图像验证等

#### `video_directory_nodes.py`
- **功能**：视频处理核心节点实现
- **主要节点**：
  - `VideoDirectoryLoader` - 视频目录加载器
  - `VideoConverterSplitter` - 视频转换分割器
  - `VideoAutoMerger` - 视频自动合并器
- **主要功能**：视频批量加载、分割、合并、格式转换、音频提取等

### 文档和示例

#### `README.md`
- **功能**：项目详细说明文档
- **内容**：节点介绍、参数说明、使用示例、功能特性等

#### `example_image_workflow.json`
- **功能**：图像批量处理示例工作流
- **内容**：展示图像目录加载、处理和条件保存的完整流程

#### `example_video_workflow.json`
- **功能**：视频批量处理示例工作流
- **内容**：展示视频分割、处理和自动合并的完整流程

### 缓存和配置

#### `.cache/`
- **功能**：自动索引缓存目录
- **主要文件**：
  - `auto_index.json` - 持久化缓存，记录处理进度和索引位置
- **用途**：确保每次运行自动加载下一个文件，实现断点续传功能

## 📁 子目录说明

### `docs/`
- **功能**：文档目录
- **内容**：项目结构说明、API文档、使用指南等

### `__pycache__/`
- **功能**：Python编译缓存目录
- **用途**：提高Python模块加载速度

## 🧪 测试文件

项目包含多个测试脚本，用于验证功能和调试：
- `test_image_validation.py` - 图像验证测试
- `test_simple_validation.py` - 简单验证测试
- `test_video_converter.py` - 视频转换测试
- `test_import.py` - 导入测试
- `test_syntax.py` - 语法测试
- `check_syntax.py` - 语法检查
- `minimal_test.py` - 最小化测试

## 🔧 核心功能模块

### 图像处理模块
- **批量加载**：支持目录递归搜索、排序、格式过滤
- **智能索引**：持久化缓存，自动加载下一个文件
- **条件保存**：根据布尔条件分类保存到不同目录
- **图像验证**：检测并跳过空图像、占位图像

### 视频处理模块
- **批量加载**：支持目录递归搜索、排序、格式过滤
- **帧提取**：使用FFmpeg提取视频帧，支持缩放和格式转换
- **音频提取**：同步提取音频数据，支持多种格式
- **视频分割**：基于关键帧的智能分割，支持自定义时长
- **自动合并**：处理完成后自动合并帧序列为视频

## 📦 依赖关系

- **核心依赖**：
  - Python 3.x
  - PyTorch
  - PIL (Pillow)
  - NumPy
  - FFmpeg (系统依赖)

- **ComfyUI集成**：
  - 符合ComfyUI自定义节点API规范
  - 无需额外安装ComfyUI之外的依赖

## 🚀 使用流程

1. **安装**：将项目目录复制到ComfyUI的`custom_nodes`目录
2. **加载**：重启ComfyUI，在节点面板中找到"目录加载与保存"分类
3. **使用**：
   - 图像处理：使用`图像目录加载器`加载图像，处理后使用`图像目录保存器`保存
   - 视频处理：使用`视频目录加载器`加载视频，处理后使用`视频自动合并器`合并
4. **批量处理**：设置好参数后，使用队列功能实现批量处理

## 📝 开发说明

- **代码风格**：遵循PEP8规范，使用中文变量名提高可读性
- **文档规范**：所有公共方法都有详细的docstring说明
- **性能优化**：使用缓存机制减少重复操作，优化文件系统访问
- **错误处理**：完善的异常处理机制，确保稳定运行

---

了解更多详细信息，请查看：
- [README.md](../README.md) - 完整项目说明
- [图像节点文档](image_nodes.md) - 图像节点详细说明
- [视频节点文档](video_nodes.md) - 视频节点详细说明
