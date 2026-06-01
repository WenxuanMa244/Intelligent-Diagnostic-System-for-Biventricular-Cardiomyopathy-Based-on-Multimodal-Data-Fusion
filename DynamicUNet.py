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
import matplotlib.pyplot as plt

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


# 计算 Dice Coefficient
def calculate_dice(preds, labels):
    preds = preds.float()
    labels = labels.float()
    intersection = (preds * labels).sum()
    dice = (2. * intersection + 1e-6) / (preds.sum() + labels.sum() + 1e-6)
    return dice.item()


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


# CBAM 模块
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
        channel_att = self.channel_attention(x)
        x = x * channel_att
        spatial_att = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_att = self.spatial_attention(spatial_att)
        x = x * spatial_att
        return x


# 定义动态 U-Net 模型
class DynamicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, filters=[32, 64, 128, 256, 512], bottleneck_factor=2):
        super(DynamicUNet, self).__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_factor = bottleneck_factor

        # Encoder
        self.encoder = nn.ModuleList()
        for i, f in enumerate(filters):
            in_ch = in_channels if i == 0 else filters[i - 1]
            self.encoder.append(self._make_conv_block(in_ch, f))

        #瓶颈层
        bottleneck_in_channels = filters[-1]
        bottleneck_out_channels = bottleneck_in_channels * bottleneck_factor
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_in_channels, bottleneck_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_out_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.ModuleList()
        reversed_filters = list(reversed(filters))
        for i, f in enumerate(reversed_filters):
            if i == 0:
                dec_in = bottleneck_out_channels
            else:
                dec_in = reversed_filters[i - 1]
            dec_out = f
            self.decoder.append(nn.ConvTranspose2d(dec_in, dec_out, kernel_size=2, stride=2))
            skip_channels = filters[len(filters) - 1 - i]
            self.decoder.append(self._make_conv_block(dec_out + skip_channels, dec_out))

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels)
        )

    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(2)(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)


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


# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 绘制 Dice 和 IoU 曲线
def plot_metrics(train_dice_scores, val_dice_scores, train_iou_scores, val_iou_scores, output_dir):
    epochs = range(1, len(train_dice_scores) + 1)

    plt.figure(figsize=(12, 5))

    # Dice Coefficient Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_dice_scores, 'b-', label='Train Dice')
    plt.plot(epochs, val_dice_scores, 'r-', label='Val Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    # IoU Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_iou_scores, 'b-', label='Train IoU')
    plt.plot(epochs, val_iou_scores, 'r-', label='Val IoU')
    plt.title('Intersection over Union (IoU)')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 本地数据集路径（替换为你的实际路径）
    train_dataset = SegmentationDataset(
        image_dir=r"C:\Users\32572\Desktop\dataset\train_label_1",
        label_mask_dir=r"C:\Users\32572\Desktop\dataset\train_label_mask_1",
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )

    val_dataset = SegmentationDataset(
        image_dir=r"C:\Users\32572\Desktop\dataset\val_oringe_1",
        label_mask_dir=r"C:\Users\32572\Desktop\dataset\val_label_mask_1",
        transform=single_channel_transform,
        label_mask_transform=label_mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=0)  
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=0)

    # 初始化 DynamicUNet 模型
    filters = [32, 64, 128, 256, 512]
    model = DynamicUNet(in_channels=1, out_channels=2, filters=filters, bottleneck_factor=2).to(device)

    # 计算模型参数量
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    num_epochs = 50
    best_val_loss = float('inf')
    save_path = 'best_model.pth'
    output_dir = 'output_images_1'

    # 用于记录指标
    train_dice_scores = []
    val_dice_scores = []
    train_iou_scores = []
    val_iou_scores = []

    # 记录总运行时间
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

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

            preds = torch.argmax(outputs, dim=1)
            train_iou += calculate_iou(preds, labels)
            train_dice += calculate_dice(preds.float(), labels.float())

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

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
                val_dice += calculate_dice(preds.float(), labels.float())

                if batch_idx % 10 == 0:
                    save_predictions(labels.cpu(), outputs.cpu(), label_paths, epoch, batch_idx,
                                     os.path.join(output_dir, 'val'))

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        # 记录指标
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)
        train_iou_scores.append(train_iou)
        val_iou_scores.append(val_iou)

        # 计算每轮时间
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}, '
              f'Epoch Time: {epoch_time:.2f}s')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    # 计算总运行时间
    total_time = time.time() - total_start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")

    # 绘制 Dice 和 IoU 曲线
    plot_metrics(train_dice_scores, val_dice_scores, train_iou_scores, val_iou_scores, output_dir)

    # 在训练完成后展示验证集的示例图像
    model.eval()
    with torch.no_grad():
        val_iter = iter(val_loader)
        images, labels, label_paths = next(val_iter)  # 获取第一个批次
        images, labels = images.to(device), labels.to(device)

        # 获取模型预测
        outputs = model(images)
        predictions = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(predictions, dim=1, keepdim=True).float()  # 预测掩码

        # 选择第一张图片进行展示
        idx = 0  # 展示第一张图片
        val_image = images[idx].cpu().numpy().squeeze()  # 原始图像 (1, 256, 256) -> (256, 256)
        val_label = labels[idx].cpu().numpy()  # 真实掩码 (256, 256)
        val_pred = predictions[idx].cpu().numpy().squeeze()  # 预测掩码 (1, 256, 256) -> (256, 256)

        # 反归一化原始图像（如果需要显示真实灰度值）
        val_image = (val_image * 0.5 + 0.5) * 255  # 从 [-1, 1] 转换回 [0, 255]
        val_image = val_image.astype(np.uint8)

        # 将掩码转换为 0-255 的范围以便可视化
        val_label = (val_label * 255).astype(np.uint8)
        val_pred = (val_pred * 255).astype(np.uint8)

        # 使用 matplotlib 显示三张图片
        plt.figure(figsize=(15, 5))

        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(val_image, cmap='gray')
        plt.title('Original Image (val_oringe)')
        plt.axis('off')

        # 真实掩码
        plt.subplot(1, 3, 2)
        plt.imshow(val_label, cmap='gray')
        plt.title('Ground Truth (val_label_mask)')
        plt.axis('off')

        # 预测掩码
        plt.subplot(1, 3, 3)
        plt.imshow(val_pred, cmap='gray')
        plt.title('Prediction (val_prediction_mask)')
        plt.axis('off')

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_sample_comparison.png'))
        plt.close()

    print(f"Validation sample comparison saved to {os.path.join(output_dir, 'val_sample_comparison.png')}")