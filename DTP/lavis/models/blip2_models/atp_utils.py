"""
Supporting file for atp.py.
"""

from enum import IntEnum
import torch
from torch import nn

import clip
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

from torchvision import models
import torch.nn.functional as F

class ModalityEmbeddingsID(IntEnum):
    TEXT_QUESTION = 0
    TEXT_EMBEDDING = 1
    TEXT_UNUSED = 2  # ignore
    VISUAL_EMBEDDING = 3
    VISUAL_UNUSED = 4  # ignore


class ModalityEmbeddings(nn.Module):
    """
    提供表示模态类型的嵌入，用于ATP模型中的多模态输入。具体用法请参见atp.py。
    """

    def __init__(self,
                 d_model: int,
                 use_text_query: bool = False,
                 use_text_cands: bool = False,
                 n_cands: int = 5):
        """
        初始化ModalityEmbeddings模块，用于处理不同模态的数据。
        这些参数的详细说明可以参考ATPConfig。

        d_model: 模型的维度（embedding的维度），即嵌入的特征维度。
        use_text_query: 是否使用文本查询嵌入，默认为False。
        use_text_cands: 是否使用文本候选嵌入，默认为False。
        n_cands: 文本候选的数量，仅在use_text_cands为True时有效，默认为5。
        """
        super().__init__()
        self.d_model = d_model  # 模型的维度（embedding的维度）

        # 创建嵌入层，用于将不同的模态ID映射到相应的嵌入向量
        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

        # 是否使用文本查询和文本候选的标志
        self.use_text_query = use_text_query
        self.use_text_cands = use_text_cands
        # 文本候选的数量，如果不使用候选文本，则n_cands为0
        self.n_cands = n_cands if use_text_cands else 0
        # 文本特征的数量，如果使用查询文本，则n_text_feats为1
        self.n_text_feats = 1 if use_text_query else 0
        # 如果使用文本候选，则将候选的数量加入到文本特征的数量中
        if use_text_cands:
            self.n_text_feats += n_cands

    def forward(self, x: torch.tensor):
        """
        前向传播方法，返回输入x的模态嵌入。

        x: 形状为(T, B, D)的tensor，其中：
            T: 序列长度（可能是视频帧数 + 文本查询 + 文本候选）
            B: 批量大小（batch_size）
            D: 输入特征维度（input_feature_dim），即每个元素的特征数

        返回:
            返回模态嵌入，形状为(T, *, D)，其中*表示可能的B维度。
        """
        T, B, D = x.size()  # 获取输入的序列长度、批量大小和特征维度
        n_frames = T - self.n_text_feats  # 获取实际的视频帧数，T减去文本特征的数量

        # 组装模态ID，用于区分不同模态（文本查询、文本候选、视觉嵌入）
        class_ids = []
        if self.use_text_query:
            # 如果使用文本查询，添加TEXT_QUESTION ID
            class_ids = [ModalityEmbeddingsID.TEXT_QUESTION, ]
        if self.use_text_cands:
            # 如果使用文本候选，添加多个TEXT_EMBEDDING ID
            class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING, ] * self.n_cands)
        # 添加视觉嵌入的ID
        class_ids.extend([ModalityEmbeddingsID.VISUAL_EMBEDDING, ] * n_frames)

        # 将class_ids转换为tensor，并设置设备和数据类型
        class_ids = torch.tensor(
            class_ids,
            dtype=torch.long,
            device=x.device
        ).unsqueeze(-1)  # 在最后一维扩展，使得形状为(T, B, 1)

        # 返回对应的模态嵌入
        return self.embedding(class_ids)  # 获取每个模态ID的嵌入向量



class VideoFeatureExtractor_ori(nn.Module):
    """
    Preprocessing module to extract features from video frames using CLIP.
    Converts video input (B, C, T, H, W) into feature sequences (N, L, D)
    compatible with ATP.
    """

    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        # Load CLIP model and preprocessing function
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.feature_dim = self.clip_model.visual.output_dim  # Feature dimension D

    def forward(self, video_frames):
        """
        Extract features from video frames.
        Args:
            video_frames (torch.Tensor): Input video, shape (B, C, T, H, W)
        Returns:
            torch.Tensor: Feature sequence, shape (B, T, D)
        """
        B, C, T, H, W = video_frames.shape  # Batch, Channels, Time, Height, Width

        # Ensure only 3 RGB channels are used(为了适应clip的维度)
        # if C > 3:
        #     video_frames = video_frames[:, :3, :, :, :]  # Retain only the first 3 channels

        # Reshape to process each frame individually (B * T, C, H, W)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        video_frames = video_frames.reshape(B * T, C, H, W)  # (B * T, C, H, W)

        # Preprocess frames using CLIP's preprocessing pipeline
        processed_frames = torch.stack([
            self.clip_preprocess(to_pil_image(frame)) for frame in video_frames
        ])  # (B * T, C, H', W')

        # Move to device
        processed_frames = processed_frames.to(self.device)

        # Extract features using CLIP's visual encoder
        with torch.no_grad():
            frame_features = self.clip_model.visual(processed_frames)  # (B * T, D)

        # Reshape back to video sequence format (B, T, D)
        frame_features = frame_features.view(B, T, self.feature_dim)  # (B, T, D)
        return frame_features

class VideoFeatureExtractor_1(nn.Module):
    """
    Preprocessing module to extract features from video frames.
    Converts video input (B, C, T, H, W) into feature sequences (N, L, D).
    """

    def __init__(self, feature_dim=768, device="cuda"):
        super().__init__()
        # Initialize feature dimension and device
        self.device = device
        self.feature_dim = feature_dim  # Feature dimension D (default 768)

        # Define a simple encoder (e.g., a lightweight CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.feature_dim),
        ).to(self.device)

    def forward(self, video_frames):
        """
        Encode video frames into feature sequences.
        Args:
            video_frames (torch.Tensor): Input video, shape (B, C, T, H, W)
        Returns:
            torch.Tensor: Feature sequence, shape (B, T, D)
        """
        B, C, T, H, W = video_frames.shape  # Batch, Channels, Time, Height, Width

        # Reshape to process each frame individually (B * T, C, H, W)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        video_frames = video_frames.reshape(B * T, C, H, W)  # (B * T, C, H, W)

        # Move frames to the appropriate device
        video_frames = video_frames.to(self.device)

        # Extract features using the internal encoder
        with torch.no_grad():
            frame_features = self.encoder(video_frames)  # (B * T, 768)

        # Reshape back to video sequence format (B, T, D)
        frame_features = frame_features.view(B, T, self.feature_dim)  # (B, T, 768)
        return frame_features


class VideoFeatureExtractor_2(nn.Module):
    def __init__(self, feature_dim=768, device="cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        # Load a pre-trained ResNet model and modify the final layer
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification head
        self.fc = nn.Linear(resnet.fc.in_features, self.feature_dim)  # Map to desired feature dimension

        self.to(self.device)

    def forward(self, video_frames):
        B, C, T, H, W = video_frames.shape

        # Reshape to process each frame individually
        video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(self.device)

        # Extract features with ResNet
        with torch.no_grad():
            features = self.encoder(video_frames)  # (B * T, 2048, 1, 1)
            features = features.view(features.size(0), -1)  # Flatten to (B * T, 2048)

        # Project to desired feature dimension
        features = self.fc(features)  # (B * T, 768)
        features = F.normalize(features, dim=-1)  # Normalize features

        # Reshape back to video sequence format
        return features.view(B, T, self.feature_dim)




class VideoFeatureExtractor_3(nn.Module):
    """
    Preprocessing module to extract features from video frames and text using CLIP.
    Converts video input (B, C, T, H, W) into feature sequences (B, T, D),
    and text input into feature vectors (B, D).
    """

    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        # Load CLIP model and preprocessing function
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.feature_dim = self.clip_model.visual.output_dim  # Feature dimension D

    def forward(self, video_frames, text_inputs):
        """
        Extract features from video frames and text.
        Args:
            video_frames (torch.Tensor): Input video, shape (B, C, T, H, W)
            text_inputs (List[str]): List of strings (questions or text queries), length B
        Returns:
            frame_features (torch.Tensor): Video frame features, shape (B, T, D)
            text_features (torch.Tensor): Text features, shape (B, D)
        """
        # Video frame feature extraction
        B, C, T, H, W = video_frames.shape  # Batch, Channels, Time, Height, Width

        # Reshape to process each frame individually (B * T, C, H, W)
        video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B * T, C, H, W)

        # Preprocess frames using CLIP's preprocessing pipeline
        processed_frames = torch.stack([
            self.clip_preprocess(to_pil_image(frame)) for frame in video_frames
        ])  # (B * T, C, H', W')

        # Move to device
        processed_frames = processed_frames.to(self.device)

        # Extract features using CLIP's visual encoder
        with torch.no_grad():
            frame_features = self.clip_model.visual(processed_frames)  # (B * T, D)

        # Reshape back to video sequence format (B, T, D)
        frame_features = frame_features.view(B, T, self.feature_dim)  # (B, T, D)

        # Text feature extraction
        text_tokens = clip.tokenize(text_inputs).to(self.device)  # Tokenize text inputs
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)  # (B, D)

        # 检查 text_features 的范围
        if torch.any(torch.isnan(text_features)):
            raise ValueError("NaN detected in text_features before normalization!")
        if torch.any(torch.isinf(text_features)):
            raise ValueError("Inf detected in text_features before normalization!")

        # Normalize text features for consistency
        text_norms = text_features.norm(dim=-1, keepdim=True)
        if torch.any(text_norms == 0):
            raise ValueError("Zero norm detected in text_features during normalization!")

        text_features = text_features / text_norms

        # 检查归一化后的范围
        if torch.any(torch.isnan(text_features)):
            raise ValueError("NaN detected in text_features after normalization!")
        if torch.any(torch.isinf(text_features)):
            raise ValueError("Inf detected in text_features after normalization!")

        return frame_features, text_features


class VideoFeatureExtractor(nn.Module):
    """
    Preprocessing module to extract features from video frames and text using CLIP.
    Converts video input (B, C, T, H, W) into feature sequences (B, T, D),
    and text input into feature vectors (B, D).
    """

    def __init__(self, model_name="ViT-B/32", device="cuda", dropout_prob=0.5):
        super().__init__()
        # Load CLIP model and preprocessing function
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.feature_dim = self.clip_model.visual.output_dim  # Feature dimension D

        # Add dropout and normalization layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.batch_norm_frame = nn.BatchNorm1d(self.feature_dim)
        self.batch_norm_text = nn.BatchNorm1d(self.feature_dim)

    def forward(self, video_frames, text_inputs):
        """
        Extract features from video frames and text.
        Args:
            video_frames (torch.Tensor): Input video, shape (B, C, T, H, W)
            text_inputs (List[str]): List of strings (questions or text queries), length B
        Returns:
            frame_features (torch.Tensor): Video frame features, shape (B, T, D)
            text_features (torch.Tensor): Text features, shape (B, D)
        """
        # Video frame feature extraction
        B, C, T, H, W = video_frames.shape  # Batch, Channels, Time, Height, Width

        # Reshape to process each frame individually (B * T, C, H, W)
        video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B * T, C, H, W)

        # Preprocess frames using CLIP's preprocessing pipeline
        processed_frames = torch.stack([
            self.clip_preprocess(to_pil_image(frame)) for frame in video_frames
        ])  # (B * T, C, H', W')

        # Move to device
        processed_frames = processed_frames.to(self.device)

        # Extract features using CLIP's visual encoder
        with torch.no_grad():
            frame_features = self.clip_model.visual(processed_frames)  # (B * T, D)

        # Reshape back to video sequence format (B, T, D)
        frame_features = frame_features.view(B, T, self.feature_dim)  # (B, T, D)

        # Apply dropout and batch normalization to frame features
        frame_features = self.dropout(frame_features)
        frame_features = self.batch_norm_frame(frame_features.transpose(1, 2)).transpose(1, 2)

        # Text feature extraction
        text_tokens = clip.tokenize(text_inputs).to(self.device)  # Tokenize text inputs
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)  # (B*T, D)

        # 检查 text_features 的范围
        if torch.any(torch.isnan(text_features)):
            raise ValueError("NaN detected in text_features before normalization!")
        if torch.any(torch.isinf(text_features)):
            raise ValueError("Inf detected in text_features before normalization!")

        # Apply dropout and batch normalization to text features
        text_features = self.dropout(text_features)
        text_features = self.batch_norm_text(text_features)

        # Normalize text features for consistency
        text_norms = text_features.norm(dim=-1, keepdim=True)
        if torch.any(text_norms == 0):
            raise ValueError("Zero norm detected in text_features during normalization!")

        text_features = text_features / text_norms

        # 检查归一化后的范围
        if torch.any(torch.isnan(text_features)):
            raise ValueError("NaN detected in text_features after normalization!")
        if torch.any(torch.isinf(text_features)):
            raise ValueError("Inf detected in text_features after normalization!")

        return frame_features, text_features