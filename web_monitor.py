import streamlit as st
import socket
import threading
import time
import sqlite3
from datetime import datetime

# 全局变量用于存储数据
blood_data = ""
heart_data = ""
tcp_client_socket = None
current_patient_id = None
current_patient_name = None

# 定义正常值范围
BLOOD_MIN = 90  # 血氧饱和度最小值
BLOOD_MAX = 100  # 血氧饱和度最大值
HEART_MIN = 50 # 心率最小值
HEART_MAX = 120 #心率最大值

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
        return True
    except:
        time.sleep(2)
        return False

# 心跳
def Ping():
    global tcp_client_socket
    try:
        keeplive = 'ping\r\n'
        tcp_client_socket.send(keeplive.encode("utf-8"))
    except:
        time.sleep(2)
        connTCP()
    # 开启定时，1秒发送一次心跳
    t = threading.Timer(1, Ping)
    t.start()

# 数据接收处理函数
def receive_data():
    global tcp_client_socket, blood_data, heart_data
    while True:
        try:
            # 接收服务器发送过来的数据
            recvData = tcp_client_socket.recv(1024)
            if len(recvData) != 0:
                data = recvData.decode('utf-8')
                # 如果收到的是心跳响应，则忽略
                if data.strip() == 'pong':
                    continue
                # 只处理包含 'msg' 的数据
                if 'msg' in data:
                    # 提取主题和msg值
                    parts = data.split('&')
                    topic = next((part.split('=')[1] for part in parts if part.startswith('topic=')), None)
                    msg = next((part.split('=')[1] for part in parts if part.startswith('msg=')), None)
                    if topic and msg:
                        if topic == 'blood004':
                            blood_data = msg
                        elif topic == 'heart004':
                            heart_data = msg
            else:
                print("连接错误，正在重新连接...")
                if connTCP():
                    Ping()
        except Exception as e:
            print("接收数据出错:", str(e))
            time.sleep(2)
            if connTCP():
                Ping()

# Streamlit界面
def save_vital_signs(patient_id, heart_rate, blood_oxygen):
    try:
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vital_signs (patient_id, heart_rate, blood_oxygen, measure_time)
            VALUES (?, ?, ?, ?)
        """, (patient_id, heart_rate, blood_oxygen, datetime.now()))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"保存数据失败: {str(e)}")

def load_patients():
    try:
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, name FROM patients")
        patients = cursor.fetchall()
        conn.close()
        return patients
    except Exception as e:
        st.error(f"加载患者列表失败: {str(e)}")
        return []

def main():
    st.set_page_config(page_title='生命体征监测系统', layout='wide')
    st.title('生命体征监测系统')
    
    # 添加患者选择下拉框
    global current_patient_id, current_patient_name
    patients = load_patients()
    patient_options = [f"{p[0]} - {p[1]}" for p in patients]
    patient_options.insert(0, "请选择患者")
    
    selected_patient = st.selectbox("选择患者", patient_options)
    
    if selected_patient != "请选择患者":
        current_patient_id = selected_patient.split(' - ')[0]
        current_patient_name = selected_patient.split(' - ')[1]
        st.success(f"当前监测患者: {current_patient_name}")
    else:
        st.warning("请先选择一个患者进行监测")
        current_patient_id = None
        current_patient_name = None
    
    # 初始化TCP连接
    if connTCP():
        # 启动x'j'n'tiao线程
        ping_thread = threading.Thread(target=Ping)
        ping_thread.daemon = True
        ping_thread.start()
        
        # 启动数据接收线程
        receive_thread = threading.Thread(target=receive_data)
        receive_thread.daemon = True
        receive_thread.start()
    
    # 创建保存按钮
    if current_patient_id:
        if st.button("保存当前数据"):
            if blood_data != "" and heart_data != "":
                try:
                    save_vital_signs(current_patient_id, float(heart_data), float(blood_data))
                    st.success("数据保存成功！")
                except Exception as e:
                    st.error(f"保存数据失败: {str(e)}")
            else:
                st.warning("暂无数据可保存")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    # 血氧数据显示
    with col1:
        st.markdown("### 血氧饱和度监测")
        st.markdown("""<style>
            .big-font {
                font-size:30px !important;
                text-align: center;
            }
            .normal {
                color: #2E7D32;
            }
            .warning {
                color: #FF0000;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0; }
                100% { opacity: 1; }
            }
            </style>""", unsafe_allow_html=True)
        blood_placeholder = st.empty()
    
    # 心率数据显示
    with col2:
        st.markdown("### 心率监测")
        st.markdown("""<style>
            .big-font {
                font-size:30px !important;
                text-align: center;
            }
            .normal {
                color: #2E7D32;
            }
            .warning {
                color: #FF0000;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0; }
                100% { opacity: 1; }
            }
            </style>""", unsafe_allow_html=True)
        heart_placeholder = st.empty()
    
    # 实时更新数据
    while True:
        with col1:
            blood_value = blood_data if blood_data else "等待数据..."
            if blood_value != "等待数据...":
                blood_value = float(blood_value)
                style_class = "normal" if BLOOD_MIN <= blood_value <= BLOOD_MAX else "warning"
                warning_text = "" if BLOOD_MIN <= blood_value <= BLOOD_MAX else "\n⚠️ 血氧饱和度异常！病人危重，请主治医生立即组织抢救！"
                blood_placeholder.markdown(
                    f'<p class="big-font {style_class}">{blood_value} %</p>'
                    f'<p style="color: red; text-align: center; font-weight: bold;">{warning_text}</p>', 
                    unsafe_allow_html=True
                )
            else:
                blood_placeholder.markdown(f'<p class="big-font normal">{blood_value}</p>', unsafe_allow_html=True)
        
        with col2:
            heart_value = heart_data if heart_data else "等待数据..."
            if heart_value != "等待数据...":
                heart_value = float(heart_value)
                style_class = "normal" if HEART_MIN <= heart_value <= HEART_MAX else "warning"
                warning_text = "" if HEART_MIN <= heart_value <= HEART_MAX else "\n⚠️ 心率异常！病人危重，请主治医生立即组织抢救！"
                heart_placeholder.markdown(
                    f'<p class="big-font {style_class}">{heart_value} BPM</p>'
                    f'<p style="color: red; text-align: center; font-weight: bold;">{warning_text}</p>', 
                    unsafe_allow_html=True
                )
                
                # 保存数据到数据库
                if current_patient_id and blood_value != "等待数据..." and heart_value != "等待数据...":
                    try:
                        save_vital_signs(current_patient_id, heart_value, blood_value)
                        print(f"数据已自动保存 - 心率: {heart_value}, 血氧: {blood_value}")
                    except Exception as e:
                        print(f"自动保存数据失败: {str(e)}")
                        st.error(f"自动保存数据失败: {str(e)}")
            else:
                heart_placeholder.markdown(f'<p class="big-font normal">{heart_value}</p>', unsafe_allow_html=True)
        
        time.sleep(1)

if __name__ == '__main__':
    main()