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
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 数据预处理变换
single_channel_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

label_mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# 确保输出和标签形状一致
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


# 定义注意力门（Attention Gate）
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # 上采样特征（gating signal）
        x1 = self.W_x(x)  # 跳跃连接特征
        psi = self.relu(g1 + x1)  # 加法融合后激活
        psi = self.psi(psi)  # 生成注意力权重
        return x * psi  # 注意力加权跳跃连接特征


# 定义 Attention U-Net 模型
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, filters=[32, 64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.filters = filters

        # 编码器
        self.encoders = nn.ModuleList()
        for i, f in enumerate(filters):
            in_ch = in_channels if i == 0 else filters[i - 1]
            self.encoders.append(self._conv_block(in_ch, f))

        # 解码器
        self.up_convs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(filters[i], filters[i - 1], kernel_size=2, stride=2))
            self.attentions.append(AttentionGate(filters[i - 1], filters[i - 1], filters[i - 1] // 2))
            self.decoders.append(self._conv_block(filters[i - 1] * 2, filters[i - 1]))

        # 输出层
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器路径
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)

        # 跳跃连接反转
        skips = skips[:-1][::-1]

        # 解码器路径
        for i, (up_conv, att, dec) in enumerate(zip(self.up_convs, self.attentions, self.decoders)):
            x = up_conv(x)  # 上采样
            skip = skips[i]  # 对应的跳跃连接
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            att_skip = att(x, skip)  # 注意力门处理跳跃连接
            x = torch.cat([x, att_skip], dim=1)  # 拼接
            x = dec(x)  # 卷积块处理

        return self.final_conv(x)


# 数据集类
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


# 保存预测结果
def save_predictions(labels, outputs, label_mask_paths, epoch, batch_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictions = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(predictions, dim=1, keepdim=True).float()

    for i in range(labels.size(0)):
        label_mask_path = label_mask_paths[i]
        prediction = predictions[i].unsqueeze(0)
        prediction_np = prediction.cpu().numpy().squeeze()
        prediction_np = (prediction_np * 255).astype(np.uint8)
        prediction_mask_pil = Image.fromarray(prediction_np, mode='L')

        original_label_mask_save_path = os.path.join(output_dir,
                                                     f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_original_label_mask.png")
        original_label_mask = Image.open(label_mask_path).convert('L')
        original_label_mask = original_label_mask.resize((256, 256))
        original_label_mask.save(original_label_mask_save_path)

        prediction_mask_save_path = os.path.join(output_dir,
                                                 f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_prediction_mask.png")
        prediction_mask_pil.save(prediction_mask_save_path)


# 训练函数的模型
def train_model(model, train_loader, val_loader, criterion, dice_loss, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    val_ious = []

    best_val_loss = float('inf')
    save_path = 'best_model.pth'
    output_dir = 'output_images'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels, _) in enumerate(
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
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

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
                iou = calculate_iou(preds, labels)
                val_iou += iou

                if batch_idx % 10 == 0:
                    save_predictions(labels.cpu(), outputs.cpu(), label_paths, epoch, batch_idx,
                                     os.path.join(output_dir, 'val'))

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, val_ious


# 绘制指标
def plot_metrics(train_losses, val_losses, val_ious):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, 'g-', label='Validation IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 设置设备
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

    # 创建数据加载器，在此处不设置多线程
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=0)

    # 初始化 Attention U-Net 模型
    filters = [32, 64, 128, 256, 512]
    model = AttentionUNet(in_channels=1, out_channels=2, filters=filters).to(device)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 训练模型
    num_epochs = 10
    train_losses, val_losses, val_ious = train_model(
        model, train_loader, val_loader, criterion, dice_loss, optimizer, scheduler, num_epochs, device
    )

    # 绘制结果
    plot_metrics(train_losses, val_losses, val_ious)