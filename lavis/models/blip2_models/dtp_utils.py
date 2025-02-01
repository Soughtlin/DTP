"""
Supporting file for atp.py.
"""

from enum import IntEnum
import clip
from torchvision.transforms.functional import to_pil_image
from dataclasses import dataclass
import math
import torch
from torch import nn


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


class VideoFeatureExtractor(nn.Module):
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


def init_weights(module):
    """自定义权重初始化函数"""
    if isinstance(module, nn.Linear):
        # 使用 Kaiming 初始化（He 初始化），适用于 ReLU 激活函数
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.TransformerEncoderLayer):
        # 遍历 TransformerEncoderLayer 中的所有子模块并初始化
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm 的权重初始化为 1，偏置为 0
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

@dataclass
class DTPConfig:
    '''
    ATPConfig contains the parameters needed for the ATPSelectorModel (and its ATPEncoder).
    '''
    # ATPEncoder params
    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 128
    d_model_ff: int = 128
    enc_dropout: float = 0.1
    use_text_query: bool = False  # at least one use_text_* needs to be true for ATP to be multimodal
    use_text_cands: bool = False  # ^ see above. (note: if both are false, ATP is vision-only)
    n_cands: int = 5  # only relevant when use_text_cands is set to true
    # ATPSelector params
    use_ste: bool = True  # controls type of selector during ATP training; see ATPSelectorModel.forward
    sel_dropout: float = 0.0
    d_input: int = 512  # size of the input vision-language embeddings (e.g. CLIP-ViT-B32 is size 512)

    @classmethod
    def from_args(cls, args):
        return cls(n_layers=args.n_layers,
                   n_heads=args.n_heads,
                   d_model=args.d_model,
                   d_model_ff=args.d_model_ff,
                   enc_dropout=args.enc_dropout,
                   use_text_query=args.use_text_query,
                   use_text_cands=args.use_text_cands,
                   n_cands=args.n_cands,
                   use_ste=args.use_ste,
                   sel_dropout=args.sel_dropout,
                   d_input=args.d_input)

class DTPEncoder(nn.Module):
    """
    ATP模型的多模态Transformer编码器。为了分析目的，ATP编码器不使用任何位置编码（没有位置编码 + Transformer / 自注意力），
    通常保持低容量。如果目标是原始准确性（而不是分析），可以放宽这些限制。
    """

    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig，包含ATP模型的（基于transformer的，非时间序列的）编码器参数。
        具体请查看ATPConfig文档。
        """
        super().__init__()
        self.d_model = config.d_model  # 模型的维度（d_model），输入和输出的特征维度
        self.dropout = nn.Dropout(p=config.enc_dropout)  # dropout层，用于防止过拟合
        # modality_encoding负责对不同模态（文本、图像等）进行嵌入
        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)

        # Transformer编码器层
        atp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,  # 输入和输出的维度
            nhead=config.n_heads,  # 注意力头的数量
            dim_feedforward=config.d_model_ff,  # 前馈层的维度
            dropout=config.enc_dropout,  # dropout率
            activation='relu'  # 激活函数
        )
        # 组成最终的Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(atp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor):
        """
        x_inputs: 输入的tensor，形状为(T, B, D)，其中：
            T: 序列长度 (frame_count)，即视频帧数
            B: 批量大小 (batch_size)
            D: 特征维度 (d_model)，即输入的每个元素的特征数量，通常和模型的隐藏层大小一致
        """
        T, B, D = x_inputs.size()  # 获取序列长度（视频帧数）、批量大小、特征维度
        assert D == self.d_model, "输入维度不匹配"  # 检查输入维度是否与模型维度匹配
        x_encoded = x_inputs * math.sqrt(self.d_model)  # 缩放输入的嵌入
        x_encoded += self.modality_encoding(x_encoded)  # 加上模态编码
        x_encoded = self.dropout(x_encoded)  # 应用dropout
        x_encoded = self.transformer_encoder(x_encoded)  # 通过Transformer编码器
        return x_encoded
