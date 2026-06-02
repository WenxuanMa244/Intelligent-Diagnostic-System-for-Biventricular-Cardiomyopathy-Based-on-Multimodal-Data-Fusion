import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# 定义单通道图像和标签掩码的预处理变换
single_channel_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道图像的归一化
])

label_mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整标签掩码大小
    transforms.ToTensor()  # 将标签掩码转换为 Tensor
])

# 确保标签和输出的形状一致
def ensure_same_shape(outputs, labels):
    if len(labels.shape) == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if outputs.shape[2:] != labels.shape[1:]:
        outputs = nn.functional.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
    return outputs, labels

# 计算 IoU
def calculate_iou(preds, labels):
    """
    计算 IoU（Intersection over Union）。
    :param preds: 预测的二值掩膜 (0 或 1)
    :param labels: 真实的二值掩膜 (0 或 1)
    :return: IoU 值
    """
    # 将 preds 和 labels 转换为整数张量
    preds = preds.long()  # 转换为 LongTensor
    labels = labels.long()  # 转换为 LongTensor

    intersection = (preds & labels).float().sum()  # 交集
    union = (preds | labels).float().sum()  # 并集
    iou = (intersection + 1e-6) / (union + 1e-6)  # 避免除以零
    return iou.item()

# 定义 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        preds = preds[:, 1, :, :]  # 只取前景类别的概率
        labels = labels.float()

        intersection = (preds * labels).sum()
        union = preds.sum() + labels.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# 定义 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 修改 conv1x1_final 的输入通道数为 out_channels * 5
        self.conv1x1_final = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.conv3x3_4(x)
        x6 = self.global_avg_pool(x)
        x6 = nn.functional.interpolate(x6, size=x.size()[2:], mode='bilinear', align_corners=False)
        # 拼接 5 个分支
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)  # 注意：这里只拼接 5 个分支
        out = self.conv1x1_final(out)
        return out

# 定义 CBAM（Convolutional Block Attention Module）
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att

        # 空间注意力
        spatial_att = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        x = x * spatial_att

        return x

# 定义 UNetLite 模型，支持多尺度特征融合和 CBAM
class UNetLite(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256, 512]):
        super(UNetLite, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        for feature in features:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                CBAM(feature)  # 添加 CBAM 模块
            ))
            in_channels = feature

        # Bottleneck with ASPP
        self.bottleneck = ASPP(features[-1], features[-1] * 2)  # 输入通道数为 features[-1]

        # Decoder path with multi-scale feature fusion
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(nn.Sequential(
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                CBAM(feature)  # 添加 CBAM 模块
            ))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoding path
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoding path with multi-scale feature fusion
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transposed convolution
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[idx + 1](concat_skip)  # Depthwise Separable Conv + CBAM block

        return self.final_conv(x)

# 定义 SegmentationDataset 类
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_mask_dir, transform=None, label_mask_transform=None):
        self.image_dir = image_dir
        self.label_mask_dir = label_mask_dir
        self.transform = transform
        self.label_mask_transform = label_mask_transform

        # 获取图像文件列表并排序
        self.image_files = sorted(os.listdir(image_dir))
        print(f"Image files: {self.image_files}")

        # 根据图像文件名生成对应的掩膜文件名列表
        # 假设图像文件名是 "image_0xxx.png"，掩膜文件名是 "mask_overlay_slice_0xxx.png"
        self.label_mask_files = [f"mask_overlay_slice_{file_name.split('_')[-1]}" for file_name in self.image_files]
        print(f"Label mask files: {self.label_mask_files}")

        # 过滤掉缺失的掩膜文件
        self.valid_indices = []
        for idx, file_name in enumerate(self.label_mask_files):
            mask_file_path = os.path.join(self.label_mask_dir, file_name)
            if os.path.exists(mask_file_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Mask file {file_name} does not exist. Skipping this file.")

        # 更新图像文件和掩膜文件列表
        self.image_files = [self.image_files[i] for i in self.valid_indices]
        self.label_mask_files = [self.label_mask_files[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像和标签掩码
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_mask_path = os.path.join(self.label_mask_dir, self.label_mask_files[idx])
        image = Image.open(image_path).convert('L')  # 确保图像始终为单通道灰度模式
        label_mask = Image.open(label_mask_path).convert('L')  # 确保标签掩码为单通道灰度模式

        # 应用图像和标签掩码的变换
        if self.transform:
            image = self.transform(image)
        if self.label_mask_transform:
            label_mask = self.label_mask_transform(label_mask)

        # 移除多余的维度
        label_mask = label_mask.squeeze(0)

        return image, label_mask, label_mask_path  # 返回图像、处理后的标签掩码和原始标签掩码路径

# 保存预测结果
def save_predictions(labels, outputs, label_mask_paths, epoch, batch_idx, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 使用 softmax 处理二分类分割
    predictions = torch.softmax(outputs, dim=1)  # 对每个像素进行 softmax
    predictions = torch.argmax(predictions, dim=1, keepdim=True).float()  # 获取每个像素的最大类别索引

    # 保存原始标签掩码和预测结果
    for i in range(labels.size(0)):
        label_mask_path = label_mask_paths[i]  # 原始标签掩码文件路径
        prediction = predictions[i].unsqueeze(0)  # 预测结果形状为 [1, H, W]

        # 将预测结果转换为 numpy 数组
        prediction_np = prediction.cpu().numpy().squeeze()  # 形状为 (H, W)

        # 将预测结果归一化到 [0, 255] 范围
        prediction_np = (prediction_np * 255).astype(np.uint8)  # 如果是二分类，prediction_np 的值为 0 或 1

        # 转换为 PIL 图像
        prediction_mask_pil = Image.fromarray(prediction_np, mode='L')  # 单通道灰度图

        # 保存原始标签掩码文件
        original_label_mask_save_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_original_label_mask.png")
        original_label_mask = Image.open(label_mask_path).convert('L')  # 确保为单通道灰度模式
        original_label_mask = original_label_mask.resize((256, 256))  # 调整大小以匹配其他图像
        original_label_mask.save(original_label_mask_save_path)

        # 保存预测结果掩码图像
        prediction_mask_save_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_prediction_mask.png")
        prediction_mask_pil.save(prediction_mask_save_path)

# 主程序
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集实例
    train_dataset = SegmentationDataset(
        image_dir=r"C:\Users\32572\Desktop\dataset\train_oringe\train_oringe_1",  # 替换为你的训练图像路径
        label_mask_dir=r"C:\Users\32572\Desktop\dataset\train_label_mask\train_label_mask_1",  # 替换为你的训练标签掩码路径
        transform=single_channel_transform,



        label_mask_transform=label_mask_transform
    )

    val_dataset = SegmentationDataset(
        image_dir=r"C:\Users\32572\Desktop\dataset\val_oringe\val_oringe_1",  # 替换为你的验证图像路径
        label_mask_dir=r"C:\Users\32572\Desktop\dataset\val_label_mask\val_label_mask_1",  # 替换为你的验证标签掩码路径
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=0)

    # 初始化模型、损失函数和优化器
    model = UNetLite(in_channels=1, out_channels=2).to(device)  # 修改为单通道输入
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 optim.Adam
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 训练配置
    num_epochs = 10
    best_val_loss = float('inf')
    save_path = 'best_model.pth'
    output_dir = 'output_images'  # 保存预测图像的目录

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 训练循环
        for batch_idx, (images, labels, label_paths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 确保标签和输出的形状一致
            outputs, labels = ensure_same_shape(outputs, labels)

            # 计算损失
            ce_loss = criterion(outputs, labels.long())  # CrossEntropyLoss 需要 (N, C, H, W) 和 (N, H, W) 形状
            dice = dice_loss(outputs, labels)
            loss = ce_loss + dice  # 结合 CrossEntropyLoss 和 Dice Loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证循环
        model.eval()
        val_loss = 0.0
        val_iou = 0.0  # 用于累计 IoU

        with torch.no_grad():
            for batch_idx, (images, labels, label_paths) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)")):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                # 确保标签和输出的形状一致
                outputs, labels = ensure_same_shape(outputs, labels)

                # 计算损失
                ce_loss = criterion(outputs, labels.long())  # CrossEntropyLoss 需要 (N, C, H, W) 和 (N, H, W) 形状
                dice = dice_loss(outputs, labels)
                loss = ce_loss + dice  # 结合 CrossEntropyLoss 和 Dice Loss
                val_loss += loss.item()

                # 计算 IoU
                preds = torch.argmax(outputs, dim=1)  # 获取预测类别 (N, H, W)
                preds = preds.long()  # 转换为 LongTensor
                labels = labels.long()  # 转换为 LongTensor
                iou = calculate_iou(preds, labels)  # 计算 IoU
                val_iou += iou

                # 保存验证集的预测结果和原始标签掩码
                if batch_idx % 10 == 0:  # 每10个批次保存一次
                    save_predictions(labels.cpu(), outputs.cpu(), label_paths, epoch, batch_idx, os.path.join(output_dir, 'val'))

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)  # 计算平均 IoU
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")