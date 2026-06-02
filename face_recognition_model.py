import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import math
import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.detect import val  # 添加YOLOv8导入
#过滤除了,pgm之外格式的数据作为数据预处理和加载
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=16):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.face_detector = YOLO('yolov8n.pt')  # 使用专门的人脸检测模型
        
        # 获取所有视频文件
        self.videos = []
        self.labels = []
        self.person_ids = []  # 存储每个视频对应的person_id
        # 创建标签映射字典
        self.label_map = {}
        current_label = 0
        for person_id in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_id)
            if os.path.isdir(person_dir):
                # 为每个文件夹分配一个数字标签
                if person_id not in self.label_map:
                    self.label_map[person_id] = current_label
                    current_label += 1
                    
                for video_file in os.listdir(person_dir):
                    if video_file.endswith(('.mp4', '.avi')):
                        self.videos.append(os.path.join(person_dir, video_file))
                        self.labels.append(self.label_map[person_id])
                        self.person_ids.append(person_id)  # 存储person_id
        
        print(f"标签映射: {self.label_map}")  # 打印标签映射关系，方便调试
    
    def detect_and_crop_face(self, image):
        # 将PIL图像转换为numpy数组
        img_array = np.array(image)
        
        # 使用YOLOv8检测人脸
        results = self.face_detector(img_array)
        
        if len(results[0].boxes) > 0:
            # 获取第一个检测到的人脸框
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 裁剪人脸区域
            face = image.crop((x1, y1, x2, y2))
            return face.resize((192, 168))  # 调整到统一大小
        return image  # 如果没检测到人脸，返回原图
    
    def calculate_optical_flow(self, current_frame, next_frame, current_person_id, target_person_id):
        """
        计算光流，只对目标人物进行光流计算
        Args:
            current_frame: 当前帧
            next_frame: 下一帧
            current_person_id: 当前视频的人物ID
            target_person_id: 目标人物ID
        """
        # 如果不是目标人物的视频，返回零光流
        if target_person_id is not None and current_person_id != target_person_id:
            zero_flow = np.zeros((current_frame.size[1], current_frame.size[0], 3), dtype=np.uint8)
            return Image.fromarray(zero_flow)
        
        # 将PIL图像转换为numpy数组
        current = np.array(current_frame)
        next_frame = np.array(next_frame)
        
        # 转换为灰度图
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            current_gray, 
            next_gray, 
            None,
            0.5,  # pyr_scale
            3,    # levels
            15,   # winsize
            3,    # iterations
            5,    # poly_n
            1.2,  # poly_sigma
            0     # flags
        )
        
        # 转换光流为RGB图像表示
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((current.shape[0], current.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # 转换回RGB
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(flow_rgb)
    #提取帧操作
    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                continue
                
            frame_count += 1
            if frame_count % 2 == 0:  # 每隔一帧采样
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                # 人脸检测和裁剪
                frame = self.detect_and_crop_face(frame)
                frames.append(frame)
                
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        try:
            video_path = self.videos[idx]
            label = self.labels[idx]
            current_person_id = self.person_ids[idx]  # 获取当前视频的person_id
            
            # 获取当前batch的第一个person_id作为目标人物
            target_person_id = self.person_ids[0] if idx == 0 else None
            
            # 提取帧序列
            frames = self.extract_frames(video_path)
            
            # 准备RGB、灰度和光流特征
            rgb_sequence = []
            gray_sequence = []
            flow_sequence = []
            
            for i in range(len(frames)-1):
                current_frame = frames[i]
                next_frame = frames[i+1]
                
                # RGB图像
                rgb_image = current_frame
                # 灰度图像
                gray_image = current_frame.convert('L')
                # 光流图像，传入person_id信息
                flow_image = self.calculate_optical_flow(current_frame, next_frame, 
                                                       current_person_id, target_person_id)
                
                if isinstance(self.transform, tuple):
                    rgb_transform, gray_transform, flow_transform = self.transform
                    rgb_image = rgb_transform(rgb_image)
                    gray_image = gray_transform(gray_image)
                    flow_image = flow_transform(flow_image)
                
                rgb_sequence.append(rgb_image)
                gray_sequence.append(gray_image)
                flow_sequence.append(flow_image)
            
            # 堆叠序列
            rgb_sequence = torch.stack(rgb_sequence)
            gray_sequence = torch.stack(gray_sequence)
            flow_sequence = torch.stack(flow_sequence)
            
            return (rgb_sequence, gray_sequence, flow_sequence), label
            
        except Exception as e:
            print(f"加载视频出错 {video_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))
    
    def __len__(self):
        return len(self.videos)
# 2. ArcFace损失函数
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)
        # 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

# 人脸识别模型（但是我在此添加了三路特征路径）
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(FaceRecognitionModel, self).__init__()
         # RGB特征提取
        self.rgb_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        
        # 灰度特征提取
        self.gray_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 光流特征提取
        self.flow_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.flow_backbone = nn.Sequential(*list(self.flow_backbone.children())[:-2])
        # 修改第一层卷积以适应单通道输入
        self.gray_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gray_backbone = nn.Sequential(*list(self.gray_backbone.children())[:-2])
        # 注意力模块
        self.rgb_channel_attention = ChannelAttention(2048)
        self.rgb_spatial_attention = SpatialAttention()
        self.gray_channel_attention = ChannelAttention(2048)
        self.gray_spatial_attention = SpatialAttention()
        # 光流注意力模块
        self.flow_channel_attention = ChannelAttention(2048)
        self.flow_spatial_attention = SpatialAttention()
        # 特征处理
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 修改特征融合层的输入维度（RGB + 灰度 + 光流层）
        self.fusion_lora = LoRALayer(6144, embedding_size)  # 2048*3
        self.fusion = nn.Sequential(
            nn.Linear(6144, embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(embedding_size, num_classes)
    
    def forward(self, rgb_x, gray_x, flow_x):
        # 处理序列数据
        batch_size, seq_len, channels, height, width = rgb_x.size()
        
        # 重塑输入以处理序列
        rgb_x = rgb_x.view(batch_size * seq_len, channels, height, width)
        gray_x = gray_x.view(batch_size * seq_len, 1, height, width)  # 灰度图是单通道
        flow_x = flow_x.view(batch_size * seq_len, channels, height, width)
        
        # RGB特征提取
        rgb_feat = self.rgb_backbone(rgb_x)
        rgb_feat = rgb_feat * self.rgb_channel_attention(rgb_feat)
        rgb_feat = rgb_feat * self.rgb_spatial_attention(rgb_feat)
        rgb_feat = self.avg_pool(rgb_feat)
        rgb_feat = rgb_feat.view(batch_size, seq_len, -1)
        rgb_feat = torch.mean(rgb_feat, dim=1)  # 对序列取平均
        
        # 灰度特征提取
        gray_feat = self.gray_backbone(gray_x)
        gray_feat = gray_feat * self.gray_channel_attention(gray_feat)
        gray_feat = gray_feat * self.gray_spatial_attention(gray_feat)
        gray_feat = self.avg_pool(gray_feat)
        gray_feat = gray_feat.view(batch_size, seq_len, -1)
        gray_feat = torch.mean(gray_feat, dim=1)  # 对序列取平均
        
        # 光流特征提取
        flow_feat = self.flow_backbone(flow_x)
        flow_feat = flow_feat * self.flow_channel_attention(flow_feat)
        flow_feat = flow_feat * self.flow_spatial_attention(flow_feat)
        flow_feat = self.avg_pool(flow_feat)
        flow_feat = flow_feat.view(batch_size, seq_len, -1)
        flow_feat = torch.mean(flow_feat, dim=1)  # 对序列取平均
        
        # 特征融合，添加残差连接
        combined_feat = torch.cat([rgb_feat, gray_feat, flow_feat], dim=1)
        lora_feat = self.fusion_lora(combined_feat)
        features = self.fusion(combined_feat) + lora_feat
        # 分类预测
        logits = self.classifier(features)
        return features, logits
# 添加LoRA模块
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = 0.1
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return self.scale * (x @ self.lora_A @ self.lora_B)

class CombinedLoss(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CombinedLoss, self).__init__()
        self.arcface = ArcFaceLoss(in_features, num_classes)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, features, logits, labels):
        arcface_output = self.arcface(features, labels)
        arcface_loss = self.ce(arcface_output, labels)  # 使用交叉熵处理ArcFace的输出
        ce_loss = self.ce(logits, labels)
        return arcface_loss + 0.5 * ce_loss

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for (rgb_inputs, gray_inputs, flow_inputs), labels in pbar:
        rgb_inputs = rgb_inputs.to(device)
        gray_inputs = gray_inputs.to(device)
        flow_inputs = flow_inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        features, logits = model(rgb_inputs, gray_inputs, flow_inputs)
        
        # 计算损失
        loss = criterion(features, logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        # 使用logits进行预测
        _, predicted = logits.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': train_loss/len(train_loader),
            'Acc': 100.*train_correct/train_total
        })
    
    return train_loss/len(train_loader), 100.*train_correct/train_total

def validate(model, val_loader, train_loader, device):
    model.eval()
    val_correct = 0
    val_total = 0
    # 首先构建训练集的特征库
    feature_bank = {}
    with torch.no_grad():
        for (rgb_inputs, gray_inputs, flow_inputs), labels in tqdm(train_loader, desc='Building feature bank'):
            features, _ = model(rgb_inputs.to(device), gray_inputs.to(device), flow_inputs.to(device))
            # 将特征按标签存储
            for feat, label in zip(features, labels):
                if label.item() not in feature_bank:
                    feature_bank[label.item()] = []
                feature_bank[label.item()].append(feat)
    
    #计算每个类别的平均特征
    for label in feature_bank:
        feature_bank[label] = torch.stack(feature_bank[label]).mean(0)
    # 验证
    with torch.no_grad():
        for (rgb_inputs, gray_inputs, flow_inputs), labels in tqdm(val_loader, desc='Validation'):
            features, _ = model(rgb_inputs.to(device), gray_inputs.to(device), flow_inputs.to(device))
            
            #计算与所有已知类别的相似度
            for feat, true_label in zip(features, labels):
                similarities = []
                for known_label, known_feat in feature_bank.items():
                    similarity = F.cosine_similarity(feat.unsqueeze(0), known_feat.unsqueeze(0))
                    similarities.append((known_label, similarity.item()))
                # 直接找到最相似的类别作为预测结果
                pred_label = max(similarities, key=lambda x: x[1])[0]
                val_correct += (pred_label == true_label.item())
                val_total += 1
    accuracy = 100. * val_correct / val_total
    return accuracy

def main():
    # 设置参数
    data_dir = r"C:\Users\32572\Desktop\person"# 自定义数据集路径在桌面上
    batch_size = 2    # 减小batch size
    num_epochs = 30     # 调整训练轮数
    learning_rate = 0.0001  # 降低学习率
    num_classes = len(os.listdir(data_dir))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    rgb_transform = transforms.Compose([
        transforms.Resize((168, 192)),  # 保持原始图像比例
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),      # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    gray_transform = transforms.Compose([
        transforms.Resize((168, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    #添加光流图像的预处理
    flow_transform = transforms.Compose([
        transforms.Resize((168, 192)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集时传入三个transform
    dataset = FaceDataset(data_dir, transform=(rgb_transform, gray_transform, flow_transform))
    
    # 获取所有可用的person_ids
    all_person_ids = list(set(dataset.person_ids))
    # 随机选择一个person_id作为测试集
    val_person_id = np.random.choice(all_person_ids)
     # 所有数据都用于训练
    train_indices = list(range(len(dataset)))  # 所有数据索引
    val_indices = train_indices  # 验证集使用相同的数据
    
    # 创建训练集和测试集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0, drop_last=True)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"验证集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试person_id: {val_person_id}")
    print(f"类别数量: {num_classes}")
    # 添加embedding_size参数
    embedding_size = 512  # 定义embedding_size
    # 创建模型
    model = FaceRecognitionModel(num_classes=num_classes,embedding_size=embedding_size)
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = CombinedLoss(embedding_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 优化学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,  # 更温和的学习率衰减
        patience=3,   # 更快响应性能下降
        verbose=True,
        min_lr=1e-6  # 设置最小学习率
    )
       # 训练和验证
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证阶段
        val_acc = validate(model, val_loader, train_loader, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Validation Acc: {val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step(1 - val_acc/100)  # 使用验证准确率的倒数作为监控指标
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_face_recognition_model.pth')
            print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')

if __name__ == '__main__':
    main()