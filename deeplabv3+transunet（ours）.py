import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import warnings
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import random

warnings.filterwarnings("ignore")

# 定义单通道图像和标签掩码的预处理变换
single_channel_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

label_mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
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
    preds = preds.long()
    labels = labels.long()
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

# 定义 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        preds = preds[:, 1, :, :]
        labels = labels.float()
        intersection = (preds * labels).sum()
        union = preds.sum() + labels.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
# 计算 Dice Coefficient
def calculate_dice(preds, labels):
    preds = preds.float()
    labels = labels.float()
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

# 定义 CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # 空间注意力
        spatial_att = torch.cat([
            torch.mean(x_channel, dim=1, keepdim=True),
            torch.max(x_channel, dim=1, keepdim=True)[0]
        ], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        x_spatial = x_channel * spatial_att
        
        # 保存注意力图用于可视化
        self.channel_att_map = channel_att.detach()
        self.spatial_att_map = spatial_att.detach()
        
        return x_spatial

# 添加热力图可视化函数
def visualize_attention_maps(model, image, output_dir, filename):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CBAM模块的注意力图
    attention_maps = []
    def hook_fn(module, input, output):
        if isinstance(module, CBAM):
            attention_maps.append({
                'channel': module.channel_att_map,
                'spatial': module.spatial_att_map
            })
    
    # 注册钩子
    hooks = []
    for module in model.modules():
        if isinstance(module, CBAM):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # 前向传播
    with torch.no_grad():
        _ = model(image)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化注意力图
    for idx, att_maps in enumerate(attention_maps):
        # 通道注意力图
        channel_att = att_maps['channel'].mean(dim=1).cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(channel_att[0], cmap='jet')
        plt.colorbar()
        plt.title(f'Channel Attention Map {idx}')
        # 空间注意力图
        spatial_att = att_maps['spatial'].squeeze().cpu().numpy()
        plt.subplot(122)
        plt.imshow(spatial_att, cmap='jet')
        plt.colorbar()
        plt.title(f'Spatial Attention Map {idx}')
        plt.savefig(os.path.join(output_dir, f'{filename}_attention_map_{idx}.png'))
        plt.close()
        # 添加热力图可视化函数

# 定义 UNetLite 解码器
class UNetLiteDecoder(nn.Module):
    def __init__(self, features=[2048, 1024, 512, 256]):
        super(UNetLiteDecoder, self).__init__()
        self.decoder = nn.ModuleList()
        self.features = features

        for i, feature in enumerate(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2 if i != 0 else feature, feature, kernel_size=2, stride=2))
            self.decoder.append(nn.Sequential(
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                CBAM(feature)
            ))
        self.final_conv = nn.Conv2d(features[-1], 2, kernel_size=1)

    def forward(self, x, skip_connections):
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        return self.final_conv(x)



class DeepLabV3Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(DeepLabV3Encoder, self).__init__()
        self.deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=False)
        if in_channels != 3:
            self.deeplabv3.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = self.deeplabv3.backbone
        
        # 修改ASPP 模块的输出通道数以匹配原始特征
        self.aspp = nn.ModuleList([
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6, bias=False),
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12, bias=False),
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18, bias=False)
        ])
        self.aspp_bn = nn.ModuleList([nn.BatchNorm2d(512) for _ in range(4)])
        self.aspp_relu = nn.ReLU(inplace=True)
        
        # 修改全局池化分支的输出通道数
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 修改最终卷积层的输入输出通道数
        self.conv1 = nn.Conv2d(2560, 2048, 1, bias=False)  # 5×512 -> 2048
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features.append(x)
        x = self.encoder.layer2(x)
        features.append(x)
        x = self.encoder.layer3(x)
        features.append(x)
        x = self.encoder.layer4(x)
        
        # 应用ASPP模块
        aspp_out = []
        for conv, bn in zip(self.aspp, self.aspp_bn):
            y = conv(x)
            y = bn(y)
            y = self.aspp_relu(y)
            aspp_out.append(y)
            
        # 全局平均池化分支
        pool = self.global_avg_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        aspp_out.append(pool)
        
        # 合并所有分支
        x = torch.cat(aspp_out, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        features.append(x)
        return x, features

# 定义完整模型
class DeepLabV3_UNetLite(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(DeepLabV3_UNetLite, self).__init__()
        self.encoder = DeepLabV3Encoder(in_channels)
        self.decoder = UNetLiteDecoder(features=[2048, 1024, 512, 256])

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections[::-1])
        return x

# 定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_mask_dir, transform=None, label_mask_transform=None):
        self.image_dir = image_dir
        self.label_mask_dir = label_mask_dir
        self.transform = transform
        self.label_mask_transform = label_mask_transform
        self.image_files = sorted(os.listdir(image_dir))
        self.label_mask_files = [f"mask_overlay_slice_{file_name.split('_')[-1]}" for file_name in self.image_files]

        self.valid_indices = []
        for idx, file_name in enumerate(self.label_mask_files):
            mask_file_path = os.path.join(self.label_mask_dir, file_name)
            if os.path.exists(mask_file_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Mask file {file_name} does not exist. Skipping this file.")

        self.image_files = [self.image_files[i] for i in self.valid_indices]
        self.label_mask_files = [self.label_mask_files[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_mask_path = os.path.join(self.label_mask_dir, self.label_mask_files[idx])
        image = Image.open(image_path).convert('L')
        label_mask = Image.open(label_mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.label_mask_transform:
            label_mask = self.label_mask_transform(label_mask)

        label_mask = label_mask.squeeze(0)
        return image, label_mask, label_mask_path

# 保存预测结果并返回图像数据
def save_predictions(labels, outputs, label_mask_paths, epoch, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictions = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(predictions, dim=1, keepdim=True).float()

    results = []
    for i in range(labels.size(0)):
        label_mask_path = label_mask_paths[i]
        prediction = predictions[i].unsqueeze(0)
        prediction_np = prediction.cpu().numpy().squeeze()
        prediction_np = (prediction_np * 255).astype(np.uint8)
        prediction_mask_pil = Image.fromarray(prediction_np, mode='L')

        # 修复原始图像路径的获取
        original_label_mask = Image.open(label_mask_path).convert('L')
        original_label_mask = original_label_mask.resize((256, 256))
        
        # 修复原始图像路径的替换逻辑
        original_image_path = label_mask_path.replace("label_mask", "oringe")
        if "mask_overlay_slice_" in label_mask_path:
            original_image_path = original_image_path.replace("mask_overlay_slice_", "slice_")
        
        try:
            original_image = Image.open(original_image_path).convert('L')
        except Exception as e:
            print(f"Error loading original image: {original_image_path}")
            print(f"Error message: {str(e)}")
            continue

        # 保存图像
        original_label_mask_save_path = os.path.join(output_dir,
                                                    f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_original_label_mask.png")
        prediction_mask_save_path = os.path.join(output_dir,
                                                f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_prediction_mask.png")
        
        original_label_mask.save(original_label_mask_save_path)
        prediction_mask_pil.save(prediction_mask_save_path)

        # 存储图像数据用于展示
        results.append({
            'original_image': np.array(original_image),
            'label_mask': np.array(original_label_mask),
            'prediction_mask': prediction_np
        })

    return results
# 主程序
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集实例（Colab 路径）
    train_dataset = SegmentationDataset(
        image_dir="/content/unzipped_folder/dataset/train_label_1",
        label_mask_dir="/content/unzipped_folder/dataset/train_label_mask_1",
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )

    val_dataset = SegmentationDataset(
        image_dir="/content/unzipped_folder/dataset/val_oringe_1",
        label_mask_dir="/content/unzipped_folder/dataset/val_label_mask_1",
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2)

    # 初始化模型
    model = DeepLabV3_UNetLite(in_channels=1, out_channels=2).to(device)



    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 训练配置
    num_epochs = 50
    best_val_loss = float('inf')
    save_path = 'best_model.pth'
    output_dir = 'output_images_2'

    # 记录指标
    val_losses = []
    val_ious = []
    epoch_times = []
    all_val_results = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        # 训练循环
        for batch_idx, (images, labels, label_paths) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs, labels = ensure_same_shape(outputs, labels)
            ce_loss = criterion(outputs, labels.long())
            dice = dice_loss(outputs, labels)
            loss = ce_loss + dice
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证循环
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0  # 添加 Dice 指标
        with torch.no_grad():
            for batch_idx, (images, labels, label_paths) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)")):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs, labels = ensure_same_shape(outputs, labels)
                ce_loss = criterion(outputs, labels.long())
                dice = dice_loss(outputs, labels)
                loss = ce_loss + dice
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_iou += calculate_iou(preds, labels)
                val_dice += calculate_dice(preds, labels)  # 计算 Dice coefficient

                # 在验证循环中添加
                if batch_idx % 10 == 0:
                    batch_results = save_predictions(labels.cpu(), outputs.cpu(), label_paths, epoch, batch_idx,
                                                os.path.join(output_dir, 'val'))
                    # 添加注意力图可视化
                    visualize_attention_maps(
                        model, 
                        images[0:1],  # 只取批次中的第一张图片
                        os.path.join(output_dir, 'attention_maps'),
                        f'epoch_{epoch}_batch_{batch_idx}'
                    )
                    all_val_results.extend(batch_results)

        # 计算平均 Dice

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)  # 计算平均 Dice
        # 记录指标
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        # 计算并记录 epoch 时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, '
              f'Epoch Time: {epoch_time:.2f}s')

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

        # 计算并打印平均 epoch 时间
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")

