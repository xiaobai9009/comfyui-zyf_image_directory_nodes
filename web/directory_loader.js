// 图像/视频目录加载器 - 目录选择按钮扩展
// 为 ImageDirectoryLoader 和 VideoDirectoryLoader 节点添加浏览目录按钮

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 节点名称映射
const NODE_NAMES = {
    IMAGE: "ImageDirectoryLoader",
    VIDEO: "VideoDirectoryLoader"
};

// 创建目录选择按钮扩展
app.registerExtension({
    name: "zyf_image_directory_nodes.directory_loader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 检查是否是目标节点
        if (![NODE_NAMES.IMAGE, NODE_NAMES.VIDEO].includes(nodeData.name)) {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this);

            // 获取目录路径输入框控件
            const getDirectoryWidget = () => {
                return (this.widgets || []).find((w) => w.name === "目录路径");
            };

            // 创建浏览按钮
            const createBrowseButton = () => {
                // 检查是否已存在浏览按钮
                if ((this.widgets || []).some((w) => w.type === "button" && w.name === "浏览目录...")) {
                    return;
                }

                // 添加浏览按钮
                this.addWidget("button", "浏览目录...", null, async () => {
                    const dirWidget = getDirectoryWidget();
                    if (!dirWidget) {
                        console.warn("未找到目录路径输入框");
                        return;
                    }

                    try {
                        // 调用后端API打开文件夹选择对话框
                        const response = await api.fetchApi("/zyf_image_directory/browse_folder", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({})
                        });

                        const data = await response.json();

                        if (!response.ok) {
                            throw new Error(data.error || "打开文件夹选择器失败");
                        }

                        if (data.cancelled) {
                            // 用户取消了选择
                            return;
                        }

                        if (data.path) {
                            // 更新目录路径输入框的值
                            dirWidget.value = data.path;
                            // 触发回调以更新节点状态
                            dirWidget.callback?.(data.path, dirWidget);
                            // 刷新画布
                            app.graph.setDirtyCanvas(true, true);
                        }
                    } catch (error) {
                        console.error("浏览目录失败:", error);
                        alert(`浏览目录失败: ${error.message}`);
                    }
                });

                // 调整节点大小以适应新按钮
                this.setSize([this.size[0], this.computeSize()[1]]);
                app.graph.setDirtyCanvas(true, true);
            };

            // 延迟添加按钮，确保所有控件已创建
            setTimeout(() => {
                createBrowseButton();
            }, 10);

            return r;
        };
    }
});
