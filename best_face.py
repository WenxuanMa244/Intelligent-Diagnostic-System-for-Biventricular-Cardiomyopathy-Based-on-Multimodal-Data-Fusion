import cv2
import torch
import numpy as np
from torchvision import transforms
from face_recognition_model import FaceRecognitionModel

def preprocess_image(image):
    # RGB图像预处理
    rgb_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((168, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    # 灰度图像预处理
    gray_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((168, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # RGB处理
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_tensor = rgb_transform(rgb_image)
    
    # 灰度处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_tensor = gray_transform(gray_image)
    
    return rgb_tensor.unsqueeze(0), gray_tensor.unsqueeze(0)

def inference_video(model_path, num_classes, source=0):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = FaceRecognitionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 初始化人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 打开视频流
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_img = frame[y:y+h, x:x+w]
            
            # 调整大小并预处理
            face_img = cv2.resize(face_img, (192, 168))
            rgb_tensor, gray_tensor = preprocess_image(face_img)
            
            # 转移到设备
            rgb_tensor = rgb_tensor.to(device)
            gray_tensor = gray_tensor.to(device)
            
            # 推理
            with torch.no_grad():
                features, logits = model(rgb_tensor, gray_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 绘制结果
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {predicted_class}', 
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                      (0, 255, 0), 2)
            cv2.putText(frame, f'Conf: {confidence:.2f}', 
                      (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                      (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Face Recognition', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 设置参数
    model_path = 'best_face_recognition_model.pth'
    num_classes = 38  # 替换为实际的类别数量
    source = 0  # 使用默认摄像头，也可以指定视频文件路径
    
    # 运行实时识别
    inference_video(model_path, num_classes, source)