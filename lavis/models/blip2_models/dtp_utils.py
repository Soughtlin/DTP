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

    def __init__(self,
                 d_model: int,
                 use_text_query: bool = False,
                 use_text_cands: bool = False,
                 n_cands: int = 5):
        """
        Initialize the ModalityEmbeddings module for handling different modalities.
        """
        super().__init__()
        self.d_model = d_model  # Model embedding dimension

        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

        self.use_text_query = use_text_query
        self.use_text_cands = use_text_cands
        self.n_cands = n_cands if use_text_cands else 0
        self.n_text_feats = 1 if use_text_query else 0
        if use_text_cands:
            self.n_text_feats += n_cands

    def forward(self, x: torch.tensor):
        """
        Forward method to return modality embeddings for input x.
        """
        T, B, D = x.size()
        n_frames = T - self.n_text_feats

        class_ids = []
        if self.use_text_query:
            class_ids = [ModalityEmbeddingsID.TEXT_QUESTION]
        if self.use_text_cands:
            class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING] * self.n_cands)
        class_ids.extend([ModalityEmbeddingsID.VISUAL_EMBEDDING] * n_frames)

        class_ids = torch.tensor(
            class_ids,
            dtype=torch.long,
            device=x.device
        ).unsqueeze(-1)

        return self.embedding(class_ids)


class VideoFeatureExtractor(nn.Module):
    """
    Extract features from video frames and text using CLIP.
    Converts video input (B, C, T, H, W) into feature sequences (B, T, D),
    and text input into feature vectors (B, D).
    """

    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.feature_dim = self.clip_model.visual.output_dim  # Feature dimension

    def forward(self, video_frames, text_inputs):

        B, C, T, H, W = video_frames.shape

        video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        processed_frames = torch.stack([
            self.clip_preprocess(to_pil_image(frame)) for frame in video_frames
        ])

        processed_frames = processed_frames.to(self.device)

        with torch.no_grad():
            frame_features = self.clip_model.visual(processed_frames)

        frame_features = frame_features.view(B, T, self.feature_dim)

        text_tokens = clip.tokenize(text_inputs).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)

        if torch.any(torch.isnan(text_features)):
            raise ValueError("NaN detected in text_features before normalization!")
        if torch.any(torch.isinf(text_features)):
            raise ValueError("Inf detected in text_features before normalization!")

        text_norms = text_features.norm(dim=-1, keepdim=True)
        if torch.any(text_norms == 0):
            raise ValueError("Zero norm detected in text_features during normalization!")

        text_features = text_features / text_norms

        if torch.any(torch.isnan(text_features)):
            raise ValueError("NaN detected in text_features after normalization!")
        if torch.any(torch.isinf(text_features)):
            raise ValueError("Inf detected in text_features after normalization!")

        return frame_features, text_features


def init_weights(module):
    
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.TransformerEncoderLayer):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

@dataclass
class DTPConfig:

    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 128
    d_model_ff: int = 128
    enc_dropout: float = 0.1
    use_text_query: bool = False  # At least one use_text_* needs to be true for DTP to be multimodal
    use_text_cands: bool = False  
    n_cands: int = 5  # Only relevant when use_text_cands is true
    use_ste: bool = True  # Controls type of selector during DTP training
    sel_dropout: float = 0.0
    d_input: int = 512  # Size of the input vision-language embeddings

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

    def __init__(self, config: DTPConfig, **kwargs):

        super().__init__()
        self.d_model = config.d_model
        self.dropout = nn.Dropout(p=config.enc_dropout)
        self.modality_encoding = ModalityEmbeddings(d_model=self.d_model,
                                                    use_text_query=config.use_text_query,
                                                    use_text_cands=config.use_text_cands,
                                                    n_cands=config.n_cands)

        dtp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(dtp_encoder_layer, config.n_layers)

    def forward(self, x_inputs: torch.tensor):

        T, B, D = x_inputs.size()
        assert D == self.d_model, "Input dimension mismatch"
        x_encoded = x_inputs * math.sqrt(self.d_model)
        x_encoded += self.modality_encoding(x_encoded)
        x_encoded = self.dropout(x_encoded)
        x_encoded = self.transformer_encoder(x_encoded)
        return x_encoded
