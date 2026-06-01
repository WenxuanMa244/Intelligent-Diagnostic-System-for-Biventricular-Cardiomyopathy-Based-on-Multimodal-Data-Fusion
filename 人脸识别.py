import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 加载固定的参考图片
ref_image_path = r"C:\Users\32572\Downloads\202110916216_compressed.jpg"
ref_image = cv2.imread(ref_image_path)
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# 检测参考图片中的人脸并提取特征点
ref_faces = face_cascade.detectMultiScale(ref_gray, scaleFactor=1.3, minNeighbors=5)
if len(ref_faces) == 0:
    raise ValueError("未在参考图片中找到人脸")

# 假设只有一张人脸
x, y, w, h = ref_faces[0]
ref_face = ref_gray[y:y + h, x:x + w]

# 使用ORB特征检测器
orb = cv2.ORB_create()
kp_ref, des_ref = orb.detectAndCompute(ref_face, None)

# 匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 打开默认摄像头（通常为0），如果有多个摄像头，请尝试更改此数字
cap = cv2.VideoCapture(0)


def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0

    for tip in finger_tips:
        tip_pos = hand_landmarks.landmark[tip]
        base_pos = hand_landmarks.landmark[tip - 2]

        if tip_pos.y < base_pos.y:
            count += 1

    return count


# 初始化语音识别器
recognizer = sr.Recognizer()


def recognize_speech():
    with sr.Microphone() as source:
        print("请说话...")
        recognizer.adjust_for_ambient_noise(source)  # 调整麦克风噪声水平
        audio = recognizer.listen(source, timeout=None)  # 录制音频

    try:
        # 使用百度语音API或其他服务进行中文语音识别
        text = recognizer.recognize_google(audio, language='zh-CN')
        print(f"你说的是: {text}")
        return text
    except sr.UnknownValueError:
        print("无法理解音频")
        return ""
    except sr.RequestError as e:
        print(f"请求错误; {e}")
        return ""


authenticated = False
auth_show_duration = 50  # 显示认证成功信息的帧数
auth_show_counter = 0
recognized_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧（可能是流结束或丢失摄像头连接）")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not authenticated:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            kp_frame, des_frame = orb.detectAndCompute(face_roi, None)

            if des_frame is not None:
                matches = bf.match(des_ref, des_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 10:  # 调整阈值以适应实际情况
                    authenticated = True
                    auth_show_counter = auth_show_duration

                match_img = cv2.drawMatches(ref_face, kp_ref, face_roi, kp_frame, matches[:10], None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow('匹配结果', match_img)

    if authenticated:
        if auth_show_counter > 0:
            cv2.putText(frame, "Authenticated", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
            auth_show_counter -= 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(hand_landmarks)
                cv2.putText(frame, f"Fingers: {finger_count}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # 实时语音识别
        recognized_text = recognize_speech()
        if recognized_text:
            cv2.putText(frame, recognized_text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('人脸识别与手势识别', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()