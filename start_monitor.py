import subprocess
import webbrowser
import time
import os

def start_server():
    # 获取当前目录下的 web_monitor.py 路径
    monitor_path = os.path.join(os.path.dirname(__file__), 'web_monitor.py')
    #启动 Streamlit 服务器
    process = subprocess.Popen(['python', '-m', 'streamlit', 'run', monitor_path])
    # 等待服务器启动
    time.sleep(3)
    # 自动打开浏览器
    webbrowser.open('http://localhost:8501')
    return process

if __name__ == '__main__':
    try:
        process = start_server()
        print("生命体征监测系统服务器已启动，按 Ctrl+C 停止...")
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\n服务器已停止")