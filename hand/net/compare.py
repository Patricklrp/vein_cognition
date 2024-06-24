import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 网络定义
class VeinNet(nn.Module):
    # 这个模型参数量很少，运算量很小，是为了方便没有GPU的同学做实验，效果未必最好，同学们可以根据自己的知识或通过学习《神经网络与深度学习》课程后优化模型
    def __init__(self):
        super(VeinNet, self).__init__()
        # 以下定义四个卷积层，作用是通过训练后其卷积核具有提取静脉特征的能力
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, groups=32)

        # 以下定义四个batch normalization层，作用是对中间数据做归一化处理
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)

        # 以下定义池化层，作用是对长和宽维度做下采样
        self.pool = nn.MaxPool2d(2, 2)

        # 以下定义激活层，作用是增加神经网络模型的非线性
        self.act = nn.LeakyReLU()

        # 以下定义最后的特征处理层，作用是将神经网络的三维矩阵特征变为一维向量特征后经过全连接层输出分类逻辑
        self.feature = nn.AdaptiveAvgPool2d(1)
        self.x2c = nn.Linear(64, 10) # 由于给的例程数据是8类，所以这里的输出维度等于类别数是8

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.act(x)
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.act(x)
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.act(x)
        # 第四层
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = self.act(x)
        # # 第五层
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.pool(x)
        # x = self.act(x)
        # 输出特征
        x = self.feature(x).view(-1, 64)
        c = self.x2c(x)
        return c,x


# 基础变量设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.getcwd() + "/model.pt"
target_path = os.getcwd() + "/base.bmp"
data_path = os.getcwd() + "/data"
train_path = os.getcwd() + "/data/train"
target_class = '73'

print(model_path)








# 深度特征维度
feature_dim = 64

# 计算余弦相似度
def cosine_similarity(x1, x2):
    return torch.matmul(x1, x2.T) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1)).unsqueeze(1)

# 深度特征匹配器类
class FeatureMatcher:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    # 提取深度特征
    def extract_features(self, image, device):
        with torch.no_grad():
            image = image.to(device)
            _, features = self.model(image)  # 获取特征层的输出
        return features


# 读入目标图片
transform = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
def read_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# 读入目标类别图片
def read_class(data_path,target_class):
    dataset = ImageFolder(root=data_path, transform=transform)
    class_to_idx = dataset.class_to_idx
    target_class_idx = class_to_idx[target_class]
    
    # 查找目标索引
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == target_class_idx]
    # 创建目标类别的子集
    target_subset = Subset(dataset, indices)
    
    data_loader = DataLoader(target_subset, batch_size=1, shuffle=False)
    return data_loader







if __name__ == '__main__':
    # 加载模型
    model = torch.load(model_path)
    
    # 读入目标图片
    target_image = read_image(target_path)
    
    # 读入目标类别图片
    data_loader = read_class(train_path,target_class)
    
    # 获取目标图片特征向量
    feature_matcher = FeatureMatcher(model) # 特征匹配类
    image_feature = feature_matcher.extract_features(target_image,device)
    # print(image_feature.shape)
    
    # 对于同类处理
    # 获取目标类图片特征向量
    features = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            _, feature = model(inputs)
            features.append(feature)
    features = torch.cat(features, dim=0)
    # print("倒数第二层特征形状:", features.shape)
    
    # 计算特征向量余弦相似度矩阵
    normalized_vector = F.normalize(image_feature, p=2, dim=1)
    normalized_matrix = F.normalize(features, p=2, dim=1)

    cosine_similarities = F.cosine_similarity(normalized_vector, normalized_matrix)
    print("same class cosine similarities:", cosine_similarities)
    print("same class average cosine similarities  = ", torch.mean(cosine_similarities))
    
    # 计算不同类之间的平均余弦相似度
    average_cosine = []
    for i in range(74,83):
        target_class = str(i)
        # 读入目标类别图片
        data_loader = read_class(train_path,target_class)
        features = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                _, feature = model(inputs)
                features.append(feature)
        features = torch.cat(features, dim=0)
        normalized_matrix = F.normalize(features, p=2, dim=1)
        cosine_similarities = F.cosine_similarity(normalized_vector, normalized_matrix)
        average_cosine.append(torch.mean(cosine_similarities))
    print ("different class average cosine similarities = ", average_cosine)


    
    



























 


    


# 计算相似度矩阵
# similarity_matrix = feature_matcher.compute_similarity_matrix(features1, features2)

# 打印相似度矩阵的形状
# print("Similarity matrix shape:", similarity_matrix.shape)

# 训练获取特征层


# 比较特征层欧式距离





