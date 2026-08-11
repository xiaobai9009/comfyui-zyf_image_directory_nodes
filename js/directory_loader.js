// 图像/视频目录加载器 - 目录选择按钮扩展
// 为 ImageDirectoryLoader 和 VideoDirectoryLoader 节点添加浏览目录按钮

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 节点名称映射
const NODE_NAMES = {
    IMAGE_LOADER: "zyf_ImageDirectoryLoader",
    VIDEO_LOADER: "zyf_VideoDirectoryLoader",
    IMAGE_SAVER: "zyf_ImageDirectorySaver",
    IMAGE_PREVIEW: "zyf_ImageSaveWithPreview",
    VIDEO_SAVER: "zyf_VideoDirectorySaver"
};

// =============================================================================
// 动态输入槽位工具函数
// 为 ImageBatchMulti 等节点提供动态增减输入端口的功能
// 添加 "Update inputs" 按钮，修改 inputcount 后点击按钮即可更新端口数量
// =============================================================================
function setupDynamicInputs(node, { type, prefix, countWidget = "inputcount", slotOptions } = {}) {
    const rebuild = () => {
        if (!node.inputs) node.inputs = [];
        const countW = node.widgets?.find(w => w.name === countWidget);
        if (!countW) return;
        const target = countW.value;
        const current = node.inputs.filter(i => i.name?.startsWith(prefix)).length;
        if (target === current) return;
        if (target < current) {
            for (let i = 0; i < current - target; i++) {
                node.removeInput(node.inputs.length - 1);
            }
        } else {
            for (let i = current + 1; i <= target; i++) {
                node.addInput(`${prefix}${i}`, type, slotOptions);
            }
        }
    };

    // 添加 "Update inputs" 按钮
    node.addWidget("button", "Update inputs", null, rebuild);

    // 监听 inputcount 控件变化（API 加载时自动触发重建）
    const countW = node.widgets?.find(w => w.name === countWidget);
    if (countW) {
        const origCb = countW.callback;
        countW.callback = function (value, canvas) {
            const r = origCb ? origCb.apply(this, arguments) : undefined;
            if (!canvas) rebuild();   // bare = API 重载；跳过交互式拖动
            return r;
        };
    }
    return rebuild;
}

// =============================================================================
// 目录选择按钮扩展
// =============================================================================
// 创建目录选择按钮扩展
app.registerExtension({
    name: "zyf_image_directory_nodes.directory_loader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // --- ImageBatchMulti 动态输入 ---
        if (nodeData.name === "zyf_ImageBatchMulti") {
            const origOnCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                origOnCreated?.apply(this);
                setupDynamicInputs(this, { type: "IMAGE", prefix: "image_", slotOptions: { shape: 7 } });
            };
        }

        // --- MaskBatchMulti 动态输入 ---
        if (nodeData.name === "zyf_MaskBatchMulti") {
            const origOnCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                origOnCreated?.apply(this);
                setupDynamicInputs(this, { type: "MASK", prefix: "mask_" });
            };
        }

        // 检查是否是目标节点
        if (![NODE_NAMES.IMAGE_LOADER, NODE_NAMES.VIDEO_LOADER, NODE_NAMES.IMAGE_SAVER, NODE_NAMES.IMAGE_PREVIEW, NODE_NAMES.VIDEO_SAVER].includes(nodeData.name)) {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this);

            // 获取目录路径输入框控件
            const getDirectoryWidget = () => {
                return (this.widgets || []).find((w) => w.name === "目录路径" || w.name === "输出目录" || w.name === "保存路径" || w.name === "输出路径");
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

// =============================================================================
// 自动排队扩展 (与 zyf-video 分段计划同款机制)
// 无需在执行队列写目录总数，前端收到后端消息后自动递增"自动索引"并排队下一个文件。
// =============================================================================
app.registerExtension({
    name: "zyf_image_directory_nodes.auto_queue_loader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 仅对图像/视频目录加载器生效
        if (![NODE_NAMES.IMAGE_LOADER, NODE_NAMES.VIDEO_LOADER].includes(nodeData.name)) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this);

            const getW = (name) => (this.widgets || []).find((w) => w.name === name);
            const startW = getW("起始索引");
            const idxW = getW("自动索引");

            // 初始化：自动索引 = 起始索引（节点创建/加载后从起始索引开始）
            if (startW && idxW) {
                idxW.value = Number(startW.value) || 0;
            }

            // 起始索引 / 目录路径 任一变化时，重置自动索引 = 起始索引
            const syncIndex = () => {
                const s = getW("起始索引");
                const i = getW("自动索引");
                if (s && i) i.value = Number(s.value) || 0;
            };
            const wrapReset = (widget) => {
                if (!widget || widget.__zyfAutoIndexWrapped) return;
                const origCb = widget.callback;
                widget.callback = function (value, ...rest) {
                    const cbRes = origCb ? origCb.apply(this, arguments) : undefined;
                    syncIndex();
                    return cbRes;
                };
                widget.__zyfAutoIndexWrapped = true;
            };
            wrapReset(startW);
            wrapReset(getW("目录路径"));

            // 顶部状态显示: 已加载数量 / 总数量 (由后端 zyf-image-status / zyf-video-status 消息实时更新)
            const statusEl = document.createElement("div");
            statusEl.className = "zyf-dir-status";
            statusEl.textContent = "0 / 0";
            Object.assign(statusEl.style, {
                textAlign: "center",
                fontSize: "13px",
                fontWeight: "bold",
                lineHeight: "20px",
                color: "var(--fg-color, #fff)",
                borderBottom: "1px solid var(--comfy-input-border, #333)",
                overflow: "hidden",
                whiteSpace: "nowrap",
            });
            const statusWidget = this.addDOMWidget("zyf_dir_status", "status", statusEl, {
                serialize: false,
                hideOnZoom: false,
                margin: 0,
            });
            statusWidget.computeSize = function (width) {
                statusEl.style.setProperty('--comfy-widget-min-height', '20px');
                statusEl.style.setProperty('--comfy-widget-height', '20px');
                return [width, 20];
            };
            // 保存引用供状态消息更新
            this._zyfStatusEl = statusEl;
            // 将状态显示移到最顶部
            if (this.widgets && this.widgets.length > 0) {
                const first = this.widgets[0];
                if (first !== statusWidget) {
                    const idx = this.widgets.indexOf(statusWidget);
                    if (idx > 0) this.widgets.splice(idx, 1);
                    this.widgets.unshift(statusWidget);
                }
            }

            // 即时刷新目录总数: 添加/改变目录后立即显示 "0 / 总数"
            const kind = nodeData.name === NODE_NAMES.VIDEO_LOADER ? "video" : "image";
            const refreshCount = () => {
                const dirW = getW("目录路径");
                const dir = dirW ? String(dirW.value || "").trim() : "";
                if (!dir) {
                    statusEl.textContent = "0 / 0";
                    app.graph.setDirtyCanvas(true, true);
                    return;
                }
                const recW = getW("递归搜索子目录");
                const extW = getW("文件扩展名过滤");
                const sortW = getW("排序方法");
                fetch("/zyf_image_directory/count", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        kind,
                        "目录路径": dir,
                        "递归搜索子目录": recW ? !!recW.value : true,
                        "文件扩展名过滤": extW ? String(extW.value || "") : "",
                        "排序方法": sortW ? String(sortW.value || "按名称") : "按名称",
                    }),
                }).then((r) => r.json()).then((d) => {
                    if (typeof d.count === "number") {
                        statusEl.textContent = `0 / ${d.count}`;
                        app.graph.setDirtyCanvas(true, true);
                    }
                }).catch(() => {});
            };
            const debouncedCount = () => {
                if (this._zyfCountTimer) clearTimeout(this._zyfCountTimer);
                this._zyfCountTimer = setTimeout(refreshCount, 400);
            };
            // 目录相关参数变化时刷新总数
            const wrapCountRefresh = (widget) => {
                if (!widget || widget.__zyfCountWrapped) return;
                const origCb = widget.callback;
                widget.callback = function (value, ...rest) {
                    const cbRes = origCb ? origCb.apply(this, arguments) : undefined;
                    debouncedCount();
                    return cbRes;
                };
                widget.__zyfCountWrapped = true;
            };
            wrapCountRefresh(getW("目录路径"));
            wrapCountRefresh(getW("递归搜索子目录"));
            wrapCountRefresh(getW("文件扩展名过滤"));
            wrapCountRefresh(getW("排序方法"));

            // 节点创建/载入后延迟一次初始刷新
            setTimeout(() => { refreshCount(); }, 60);

            return r;
        };
    },

    async init() {
        const zyfDirPromptCache = new Map(); // nodeId -> { prompt, timestamp }
        const zyfDirTargetIndex = new Map(); // nodeId -> 已排队的目标索引(单调递增, 防止重复/回退排队)

        // 拦截 /prompt 请求，捕获并缓存加载器节点的工作流 prompt 模板
        const originalFetch = window.fetch;
        window.fetch = function (...args) {
            try {
                const url = args[0];
                const options = args[1] || {};
                if (url && (url.includes('/prompt') || url.endsWith('/prompt')) &&
                    (!options.method || String(options.method).toUpperCase() === 'POST')) {
                    if (options.body) {
                        const body = typeof options.body === 'string' ? JSON.parse(options.body) : options.body;
                        if (body && body.prompt && typeof body.prompt === 'object') {
                            for (const nodeId in body.prompt) {
                                const nodeData = body.prompt[nodeId];
                                const ct = String(nodeData?.class_type || '');
                                if (ct === NODE_NAMES.IMAGE_LOADER || ct === NODE_NAMES.VIDEO_LOADER) {
                                    zyfDirPromptCache.set(nodeId, {
                                        prompt: JSON.parse(JSON.stringify(body.prompt)),
                                        timestamp: Date.now()
                                    });
                                    // 新的一轮手动运行: 自动索引控件值 = 起始索引(自动排队不改控件值)。
                                    // 此时重置单调递增去重, 保证"中断后重跑从头开始"仍能正常排队。
                                    const inputs = (nodeData && nodeData.inputs) || {};
                                    const ai = Number(inputs["自动索引"]);
                                    const si = Number(inputs["起始索引"] ?? 0);
                                    if (ai <= si) {
                                        zyfDirTargetIndex.delete(nodeId);
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (e) {}
            return originalFetch.apply(this, args);
        };

        function updateAutoIndexInPrompt(promptObj, nodeId, nextIndex) {
            const targetIdStr = String(nodeId);
            for (const nid in promptObj) {
                if (String(nid) === targetIdStr) {
                    const nodeData = promptObj[nid];
                    nodeData.inputs = nodeData.inputs || {};
                    // 无论该字段是否由 INPUT_TYPES 作为控件序列化，都直接写入/覆盖
                    nodeData.inputs["自动索引"] = nextIndex;
                    return true;
                }
            }
            return false;
        }

        function queueNextWithPrompt(targetNodeId, nextIndex) {
            const targetIdStr = String(targetNodeId);
            const cached = zyfDirPromptCache.get(targetIdStr);
            if (cached && cached.prompt) {
                const promptToSend = JSON.parse(JSON.stringify(cached.prompt));
                if (updateAutoIndexInPrompt(promptToSend, targetIdStr, nextIndex)) {
                    fetch("/prompt", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ prompt: promptToSend }),
                    }).then((r) => {
                        if (r.ok) {
                            zyfDirPromptCache.set(targetIdStr, {
                                prompt: JSON.parse(JSON.stringify(promptToSend)),
                                timestamp: Date.now()
                            });
                        } else {
                            console.error(`[zyf_image_directory] 自动排队失败: ${r.status}`);
                        }
                    }).catch((e) => {
                        console.error("[zyf_image_directory] 自动排队请求错误:", e);
                    });
                } else {
                    console.warn(`[zyf_image_directory] prompt 中未找到自动索引字段（节点 ${targetIdStr}）`);
                }
            } else {
                console.warn(`[zyf_image_directory] 无缓存 prompt，无法自动排队（节点 ${targetIdStr}）`);
            }
        }

        function handleAutoQueue(event) {
            const { uid, next_index } = event.detail || {};
            if (uid == null || next_index == null) return;
            const targetIdStr = String(uid);
            // 单调递增去重: 只排队比已排队索引更大的索引, 忽略过期/重复消息
            const last = zyfDirTargetIndex.get(targetIdStr);
            if (last != null && next_index <= last) {
                return;
            }
            zyfDirTargetIndex.set(targetIdStr, next_index);
            queueNextWithPrompt(uid, next_index);
        }

        api.addEventListener("zyf-image-auto-queue", handleAutoQueue);
        api.addEventListener("zyf-video-auto-queue", handleAutoQueue);

        // 接收后端状态消息，实时更新插件顶部的状态显示 (已加载数量 / 总数量)
        function handleStatus(event) {
            const { uid, total, loaded } = event.detail || {};
            if (uid == null) return;
            const node = app.graph._nodes.find((n) => String(n.id) === String(uid));
            if (node && node._zyfStatusEl) {
                node._zyfStatusEl.textContent = `${loaded} / ${total}`;
                app.graph.setDirtyCanvas(true, true);
            }
        }
        api.addEventListener("zyf-image-status", handleStatus);
        api.addEventListener("zyf-video-status", handleStatus);
    }
});

// =============================================================================
// 视频目录保存器扩展: 编码格式切换时自动调整 CRF 默认值 (H264=19, H265=22)
// =============================================================================
app.registerExtension({
    name: "zyf_image_directory_nodes.video_directory_saver",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAMES.VIDEO_SAVER) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this);

            const getW = (name) => (this.widgets || []).find((w) => w.name === name);
            const codecW = getW("编码格式");
            const crfW = getW("质量CRF");
            if (codecW && crfW) {
                const origCb = codecW.callback;
                codecW.callback = function (value, ...rest) {
                    const cbRes = origCb ? origCb.apply(this, arguments) : undefined;
                    // 仅当 CRF 处于另一编码的默认值时, 才切换为新编码的默认值, 避免覆盖用户自定义值
                    const curCrf = Number(crfW.value);
                    if (String(value) === "H265" && curCrf === 19) {
                        crfW.value = 22;
                    } else if (String(value) === "H264" && curCrf === 22) {
                        crfW.value = 19;
                    }
                    app.graph.setDirtyCanvas(true, true);
                    return cbRes;
                };
            }

            return r;
        };
    }
});

// =============================================================================
// 工具函数: widget 视觉显隐 (复刻 zyf-video 的 hideWidgetVisually / applyWidgetVisibility)
// =============================================================================
function enableHiddenTypeToggle(widget) {
    if (!widget || widget._zyfHiddenTypeToggle) return;
    widget._zyfHiddenTypeToggle = true;
    widget._zyfOriginalType = widget.type;
    Object.defineProperty(widget, "type", {
        get() {
            if (widget.hidden && !widget._zyfKeepTypeOnHide) return "hidden";
            return widget._zyfOriginalType;
        },
        set(value) {
            widget._zyfOriginalType = value;
        },
    });
}

function hideWidgetVisually(widget) {
    if (!widget) return;
    enableHiddenTypeToggle(widget);
    widget.hidden = widget.hidden ?? true;
    if (!widget._zyfOriginalComputeSize) {
        widget._zyfOriginalComputeSize = widget.computeSize?.bind(widget);
    }
    if (!widget._zyfOriginalDraw) {
        widget._zyfOriginalDraw = widget.draw?.bind(widget);
    }
    if (!widget._zyfOriginalMouse) {
        widget._zyfOriginalMouse = widget.mouse?.bind(widget);
    }
    widget.computeSize = (target_width) => {
        if (widget.hidden) return [0, 0];
        if (widget._zyfOriginalComputeSize) return widget._zyfOriginalComputeSize(target_width);
        return [target_width, LiteGraph.NODE_WIDGET_HEIGHT];
    };
    if (widget._zyfOriginalDraw) {
        widget.draw = (ctx, node, widget_width, y, widget_height) => {
            if (widget.hidden) return;
            return widget._zyfOriginalDraw(ctx, node, widget_width, y, widget_height);
        };
    }
    if (widget._zyfOriginalMouse) {
        widget.mouse = function () {
            if (widget.hidden) return true;
            return widget._zyfOriginalMouse(...arguments);
        };
    }
    widget.serialize = true;
    applyWidgetVisibility(widget);
}

function applyWidgetVisibility(widget) {
    if (!widget) return;
    const el = widget.inputEl || widget.input || widget.el || widget.element;
    if (!el || !el.style) return;
    if (widget.hidden) {
        el.style.display = "none";
        el.style.pointerEvents = "none";
        el.style.height = "0px";
        el.style.width = "0px";
    } else {
        el.style.display = "";
        el.style.pointerEvents = "";
        el.style.height = "";
        el.style.width = "";
    }
}

// =============================================================================
// 强制尺寸子选项显隐逻辑 (复刻 zyf-video 的 updateCustomSizeLogic)
// 规则:
//   - 禁用/Disabled              : 全部隐藏
//   - 自定义短边                 : 显示 自定义短边 + 图像尺寸倍数
//   - 自定义长边                 : 显示 自定义长边 + 图像尺寸倍数
//   - 自定义宽高                 : 显示 自定义宽度 + 自定义高度 + 图像尺寸倍数
//   - 预设 (480x?, ?x480, ...)   : 仅显示 图像尺寸倍数
// =============================================================================
function updateCustomSizeLogic(sizeWidget, customShortWidget, customLongWidget, customWidthWidget, customHeightWidget, multipleWidget) {
    const sz = String(sizeWidget.value || "");
    let showShort = false;
    let showLong = false;
    let showWidth = false;
    let showHeight = false;
    let showMultiple = true;
    switch (sz) {
        case "自定义长边":
            showLong = true;
            break;
        case "自定义短边":
            showShort = true;
            break;
        case "自定义宽高":
            showWidth = true;
            showHeight = true;
            break;
        case "禁用":
        case "Disabled":
            showMultiple = false;
            break;
        default:
            // Preset sizes (480x?, ?x480, 480x480, 832x?, ?x832, 832x832)
            break;
    }
    if (customShortWidget) {
        customShortWidget.hidden = !showShort;
        applyWidgetVisibility(customShortWidget);
    }
    if (customLongWidget) {
        customLongWidget.hidden = !showLong;
        applyWidgetVisibility(customLongWidget);
    }
    if (customWidthWidget) {
        customWidthWidget.hidden = !showWidth;
        applyWidgetVisibility(customWidthWidget);
    }
    if (customHeightWidget) {
        customHeightWidget.hidden = !showHeight;
        applyWidgetVisibility(customHeightWidget);
    }
    if (multipleWidget) {
        multipleWidget.hidden = !showMultiple;
        applyWidgetVisibility(multipleWidget);
    }
}

// =============================================================================
// 视频目录加载器扩展: 强制尺寸下拉框切换子选项显隐
// =============================================================================
app.registerExtension({
    name: "zyf_image_directory_nodes.video_loader_force_size",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAMES.VIDEO_LOADER) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this);
            const getW = (name) => (this.widgets || []).find((w) => w.name === name);

            const sizeWidget = getW("强制尺寸");
            const customShortWidget = getW("自定义短边");
            const customLongWidget = getW("自定义长边");
            const customWidthWidget = getW("自定义宽度");
            const customHeightWidget = getW("自定义高度");
            const multipleWidget = getW("图像尺寸倍数");

            if (sizeWidget !== undefined) {
                // 初始为子 widget 注册显隐包装
                [customShortWidget, customLongWidget, customWidthWidget, customHeightWidget, multipleWidget]
                    .forEach((w) => w && hideWidgetVisually(w));

                // 初始化默认值 (防止 null/undefined)
                if (customShortWidget && (customShortWidget.value === null || customShortWidget.value === undefined)) {
                    customShortWidget.value = customShortWidget.options?.default ?? 512;
                }
                if (customLongWidget && (customLongWidget.value === null || customLongWidget.value === undefined)) {
                    customLongWidget.value = customLongWidget.options?.default ?? 832;
                }
                if (customWidthWidget && (customWidthWidget.value === null || customWidthWidget.value === undefined)) {
                    customWidthWidget.value = customWidthWidget.options?.default ?? 480;
                }
                if (customHeightWidget && (customHeightWidget.value === null || customHeightWidget.value === undefined)) {
                    customHeightWidget.value = customHeightWidget.options?.default ?? 480;
                }
                if (multipleWidget && (multipleWidget.value === null || multipleWidget.value === undefined)) {
                    multipleWidget.value = multipleWidget.options?.default ?? "无";
                }

                const node = this;
                // 下拉框变化 → 更新子选项显隐 + 调整节点高度
                const origCb = sizeWidget.callback;
                sizeWidget.callback = function (value, ...rest) {
                    const cbRes = origCb ? origCb.apply(this, arguments) : undefined;
                    updateCustomSizeLogic(sizeWidget, customShortWidget, customLongWidget, customWidthWidget, customHeightWidget, multipleWidget);
                    if (node.setSize) {
                        node.setSize([node.size[0], node.computeSize()[1]]);
                    }
                    app.graph.setDirtyCanvas(true, true);
                    return cbRes;
                };

                // 初次加载也刷新一次 (确保从工作流恢复后显隐正确)
                updateCustomSizeLogic(sizeWidget, customShortWidget, customLongWidget, customWidthWidget, customHeightWidget, multipleWidget);
                if (this.setSize) {
                    this.setSize([this.size[0], this.computeSize()[1]]);
                }
                app.graph.setDirtyCanvas(true, true);
            }

            return r;
        };
    }
});
