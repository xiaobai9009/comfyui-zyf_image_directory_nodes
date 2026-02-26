# 图像/视频目录加载器 - 后端API服务
# 提供文件夹选择对话框等前端功能支持

import os
import sys
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor
import server
from aiohttp import web

# 线程池用于执行阻塞操作
_executor = ThreadPoolExecutor(max_workers=2)


def open_folder_dialog():
    """打开原生文件夹选择对话框并返回选择的路径。"""
    
    # Windows: 使用 PowerShell 打开文件夹选择对话框
    if sys.platform == 'win32':
        try:
            powershell_script = '''
            Add-Type -AssemblyName System.Windows.Forms
            
            $dialog = New-Object System.Windows.Forms.OpenFileDialog
            $dialog.ValidateNames = $false
            $dialog.CheckFileExists = $false
            $dialog.CheckPathExists = $true
            $dialog.Title = "选择目录"
            $dialog.FileName = "文件夹选择"
            $dialog.Filter = "文件夹|*.folder"
            
            # 创建父窗体确保对话框置顶
            $form = New-Object System.Windows.Forms.Form
            $form.TopMost = $true
            $form.MinimizeBox = $false
            $form.MaximizeBox = $false
            $form.WindowState = [System.Windows.Forms.FormWindowState]::Minimized
            $form.ShowInTaskbar = $false
            $form.Show()
            $form.Activate()
            
            $result = $dialog.ShowDialog($form)
            $form.Close()
            
            if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
                $folderPath = Split-Path -Parent $dialog.FileName
                Write-Output $folderPath
            } else {
                Write-Output "::CANCELLED::"
            }
            '''
            
            # 使用隐藏窗口运行 PowerShell
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            
            result = subprocess.run(
                ['powershell', '-ExecutionPolicy', 'Bypass', '-NoProfile', '-Command', powershell_script],
                capture_output=True,
                text=True,
                startupinfo=startupinfo,
                timeout=120  # 2分钟超时
            )
            
            folder_path = result.stdout.strip()
            
            if folder_path == "::CANCELLED::" or not folder_path:
                return {"cancelled": True}
            
            if os.path.isdir(folder_path):
                return {"path": os.path.normpath(folder_path), "cancelled": False}
            else:
                return {"cancelled": True}
                
        except subprocess.TimeoutExpired:
            return {"error": "文件夹选择对话框超时，请重试。"}
        except Exception as e:
            print(f"PowerShell 文件夹对话框失败: {e}")
            # 回退到 tkinter
    
    # 尝试使用 tkinter (跨平台)
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        # 确保对话框置顶
        root.wm_attributes('-topmost', 1)
        root.focus_force()
        
        # Windows 额外处理
        if sys.platform == 'win32':
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
        
        folder_path = filedialog.askdirectory(
            parent=root,
            title="选择目录",
            mustexist=True
        )
        
        root.destroy()
        
        if folder_path:
            return {"path": os.path.normpath(folder_path), "cancelled": False}
        else:
            return {"cancelled": True}
            
    except ImportError:
        pass
    except Exception as tk_error:
        print(f"Tkinter 文件夹对话框失败: {tk_error}")
    
    # Linux/Mac: 尝试 zenity 或 kdialog
    if sys.platform != 'win32':
        # 尝试 zenity (GNOME 桌面环境)
        try:
            result = subprocess.run(
                ['zenity', '--file-selection', '--directory', '--title=选择目录'],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                folder_path = result.stdout.strip()
                if folder_path and os.path.isdir(folder_path):
                    return {"path": os.path.normpath(folder_path), "cancelled": False}
            return {"cancelled": True}
        except FileNotFoundError:
            pass
        except Exception:
            pass
        
        # 尝试 kdialog (KDE 桌面环境)
        try:
            result = subprocess.run(
                ['kdialog', '--getexistingdirectory', '--title', '选择目录'],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                folder_path = result.stdout.strip()
                if folder_path and os.path.isdir(folder_path):
                    return {"path": os.path.normpath(folder_path), "cancelled": False}
            return {"cancelled": True}
        except FileNotFoundError:
            pass
        except Exception:
            pass
    
    return {"error": "无法打开文件夹选择对话框。请安装 tkinter 或手动输入路径。"}


# 注册API路由
@server.PromptServer.instance.routes.post("/zyf_image_directory/browse_folder")
async def browse_folder(request):
    """打开原生文件夹选择对话框并返回选择的路径。"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, open_folder_dialog)
        
        if "error" in result:
            return web.json_response({"error": result["error"]}, status=500)
        
        return web.json_response(result)
        
    except Exception as e:
        import traceback
        print(f"browse_folder 错误: {traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)


print("[zyf_image_directory_nodes] 服务器扩展已加载")
