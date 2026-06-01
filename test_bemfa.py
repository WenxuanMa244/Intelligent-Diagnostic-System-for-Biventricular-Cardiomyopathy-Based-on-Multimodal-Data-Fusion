import socket
import threading
import time

def connTCP():
    global tcp_client_socket
    # 创建socket
    tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # IP 和端口
    server_ip = 'bemfa.com'
    server_port = 8344
    try:
        # 连接服务器
        tcp_client_socket.connect((server_ip, server_port))
        # 发送订阅指令
        substr_blood = 'cmd=3&uid=fe15c152f1c44192b44a399cb71ea8f9&topic=blood004\r\n'
        tcp_client_socket.send(substr_blood.encode("utf-8"))
        substr_heart = 'cmd=3&uid=fe15c152f1c44192b44a399cb71ea8f9&topic=heart004\r\n'
        tcp_client_socket.send(substr_heart.encode("utf-8"))
    except:
        time.sleep(2)
        connTCP()

# 心跳
def Ping():
    try:
        keeplive = 'ping\r\n'
        tcp_client_socket.send(keeplive.encode("utf-8"))
    except:
        time.sleep(2)
        connTCP()
    # 开启定时，1秒发送一次心跳
    t = threading.Timer(1, Ping)
    t.start()   

connTCP()
Ping()

while True:
    try:
        # 接收服务器发送过来的数据
        recvData = tcp_client_socket.recv(1024)
        if len(recvData) != 0:
            data = recvData.decode('utf-8')
            # 如果收到的是心跳响应，则忽略
            if data.strip() == 'pong':
                continue
            # 只输出包含 'msg' 的数据
            if 'msg' in data:
                # 提取主题和msg值
                parts = data.split('&')
                topic = next((part.split('=')[1] for part in parts if part.startswith('topic=')), None)
                msg = next((part.split('=')[1] for part in parts if part.startswith('msg=')), None)
                if topic and msg:
                    print(f'{topic}={msg}')
        else:
            print("连接错误，正在重新连接...")
            connTCP()
    except Exception as e:
        print("接收数据出错:", str(e))
        time.sleep(2)
        connTCP()