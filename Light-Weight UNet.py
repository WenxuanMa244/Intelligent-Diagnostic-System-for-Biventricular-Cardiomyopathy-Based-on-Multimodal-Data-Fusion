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


# 定义 ChannelPool 类
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# 定义 Conv 类
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 定义 Residual 类
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


# 定义 PAM_Module 类
class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


# 定义 CAM_Module 类
class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class DDA(nn.Module):
    def __init__(self, ch, r_2, drop_rate=0.2):
        super(DDA, self).__init__()
        ch_1 = ch_2 = ch_int = ch_out = ch  # 确保输出通道数与 ASPP 一致

        # Channel attention
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # Bilinear modelling
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        # Position attention (PAM)
        self.pam = PAM_Module(ch_int)
        # Channel attention (CAM)
        self.cam = CAM_Module(ch_int)

    def forward(self, g, x):
        # Bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # Channel attention
        g_in = g
        g_in = self.cam(g_in)
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # Position attention
        x_in = x
        x_in = self.pam(x_in)
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in

        # Residual fusion
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


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
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
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
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.conv1x1_final(out)
        return out


class DDA_ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18], reduction=16):
        super(DDA_ASPP, self).__init__()
        self.dda = DDA(in_channels, reduction)
        self.aspp = ASPP(in_channels, out_channels, rates)

        # 修改 fusion_conv 的输入通道数为 out_channels + out_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        dda_out = self.dda(x, x)
        aspp_out = self.aspp(x)

        # 确保 dda_out 和 aspp_out 的通道数一致
        if dda_out.shape[1] != aspp_out.shape[1]:
            dda_out = nn.Conv2d(dda_out.shape[1], aspp_out.shape[1], kernel_size=1).to(dda_out.device)(dda_out)

        fused_out = torch.cat([dda_out, aspp_out], dim=1)  # 在通道维度上拼接
        fused_out = self.fusion_conv(fused_out)
        return fused_out


# 定义 UNetLite 模型
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

        # Bottleneck with DDA_ASPP
        self.bottleneck = DDA_ASPP(features[-1], features[-1] * 2)

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
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

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
        self.label_mask_files = [f"mask_overlay_slice_{file_name.split('_')[-1]}" for file_name in self.image_files]

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
        image = Image.open(image_path).convert('L')  # 确保图像为单通道灰度模式
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
    os.makedirs(output_dir, exist_ok=True)

    # 使用 softmax 处理二分类分割
    predictions = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(predictions, dim=1, keepdim=True).float()

    # 保存原始标签掩码和预测结果
    for i in range(labels.size(0)):
        label_mask_path = label_mask_paths[i]
        prediction = predictions[i].unsqueeze(0)

        # 将预测结果转换为 numpy 数组
        prediction_np = prediction.cpu().numpy().squeeze()
        prediction_np = (prediction_np * 255).astype(np.uint8)

        # 转换为 PIL 图像
        prediction_mask_pil = Image.fromarray(prediction_np, mode='L')

        # 保存原始标签掩码文件
        original_label_mask_save_path = os.path.join(output_dir,
                                                     f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_original_label_mask.png")
        original_label_mask = Image.open(label_mask_path).convert('L')
        original_label_mask = original_label_mask.resize((256, 256))
        original_label_mask.save(original_label_mask_save_path)

        # 保存预测结果掩码图像
        prediction_mask_save_path = os.path.join(output_dir,
                                                 f"epoch_{epoch}_batch_{batch_idx}_sample_{i}_prediction_mask.png")
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
    model = UNetLite(in_channels=1, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        for batch_idx, (images, labels, label_paths) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 确保标签和输出的形状一致
            outputs, labels = ensure_same_shape(outputs, labels)

            # 计算损失
            ce_loss = criterion(outputs, labels.long())
            dice = dice_loss(outputs, labels)
            loss = ce_loss + dice
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证循环
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for batch_idx, (images, labels, label_paths) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)")):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                # 确保标签和输出的形状一致
                outputs, labels = ensure_same_shape(outputs, labels)

                # 计算损失
                ce_loss = criterion(outputs, labels.long())
                dice = dice_loss(outputs, labels)
                loss = ce_loss + dice
                val_loss += loss.item()

                # 计算 IoU
                preds = torch.argmax(outputs, dim=1)
                preds = preds.long()
                labels = labels.long()
                iou = calculate_iou(preds, labels)
                val_iou += iou

                # 保存验证集的预测结果和原始标签掩码
                if batch_idx % 10 == 0:
                    save_predictions(labels.cpu(), outputs.cpu(), label_paths, epoch, batch_idx,
                                     os.path.join(output_dir, 'val'))

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        # 更新学习率
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")