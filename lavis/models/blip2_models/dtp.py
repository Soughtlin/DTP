from lavis.models.blip2_models.dtp_utils import *
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from dtp_utils import *


def Discriminator(frames: torch.Tensor, segment_length: int, threshold: float) -> torch.Tensor:
    """
    Compute the similarity score between adjacent frames in each segment and return 1 or 0 based on the threshold.
    Args:
        frames (torch.Tensor): Input frames with shape (B, T, N, C).
        segment_length (int): Number of frames in each segment.
        threshold (float): Similarity threshold in the range (0, 1).

    Returns:
        torch.Tensor: Similarity judgment for each segment with shape (B, T-segment_length+1, N), values are 0 or 1.
    """
    B, T, N, C = frames.shape
    assert segment_length <= T, "Segment length must be less than or equal to T."

    result = torch.zeros((B, T - segment_length + 1, N), device=frames.device)

    for start in range(T - segment_length + 1):
        segment = frames[:, start:start + segment_length, :, :]
        similarity_matrix = F.cosine_similarity(segment[:, :-1, :], segment[:, 1:, :], dim=-1)
        segment_similarity = similarity_matrix.mean(dim=1)

        normalized_similarity = (segment_similarity - segment_similarity.min()) / (segment_similarity.max() - segment_similarity.min())
        result[:, start, :] = (normalized_similarity > threshold).float()

    return result


class SelectorModel(nn.Module):
    """
    DTP Selector Model. Given image-text encoded sequences, outputs a discrete selection for the input frames.
    """

    def __init__(self, config: DTPConfig, **kwargs):
        """
        config: DTPConfig to initialize DTPSelectorModel and its encoder.
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_input, config.d_model) 
        self.dtp_encoder = DTPEncoder(config, **kwargs)  
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)  

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_query: Optional[torch.tensor] = None,
                x_txt_cands: Optional[torch.tensor] = None,
                **kwargs):
        """
        Perform DTP selection. Input image-text encoded sequences and return the selected (unmodified) visual embeddings and selection mask.
        """
        B, T, D = x_vis_seq.size()  # Get batch size, frame count, feature dimension
        x_vis_seq = x_vis_seq.permute(1, 0, 2)  # (T, B, D)

        n_text_feats = self.dtp_encoder.modality_encoding.n_text_feats  

        x_inputs = []
        if self.config.use_text_query:
            assert x_txt_query is not None, "x_txt_query is required."
            x_inputs.append(x_txt_query.unsqueeze(0)) 
        if self.config.use_text_cands:
            assert x_txt_cands is not None, "x_txt_cands is required."
            x_inputs.append(x_txt_cands.permute(1, 0, 2))  
        x_inputs.append(x_vis_seq) 
        x_inputs = torch.cat(x_inputs, dim=0)  

        x_encoded = self.embedding(self.dropout(x_inputs))  
        x_dtp_encoded = self.dtp_encoder(x_encoded)[n_text_feats:] 

        x_logits = self.logits(self.dropout(x_dtp_encoded)) 

        if torch.any(torch.isnan(x_logits)):
            raise ValueError("NaN detected in x_logits.")
        if torch.any(torch.isinf(x_logits)):
            raise ValueError("Inf detected in x_logits.")

        if self.training:
            if self.config.use_ste:
                selection_mask = F.gumbel_softmax(x_logits, dim=0, hard=True)  
            else:
                temperature = kwargs.get("temperature", 1.0)
                if temperature <= 0:
                    raise ValueError(f"Invalid temperature: {temperature}")
                selection_mask = F.softmax(x_logits / temperature, dim=0)  
        else:
            selection_index_argmax = x_logits.max(dim=0, keepdim=True)[1] 
            selection_mask = torch.zeros_like(x_logits).scatter_(
                dim=0, index=selection_index_argmax, value=1.0
            )  # Set the selected index to 1

        if torch.any(torch.isnan(selection_mask)):
            raise ValueError("NaN detected in selection_mask.")
        if torch.any(torch.isinf(selection_mask)):
            raise ValueError("Inf detected in selection_mask.")

        selected_frames = (selection_mask * x_vis_seq).sum(dim=0) 
                    
        if torch.any(torch.isnan(selected_frames)):
            raise ValueError("NaN detected in selected_frames.")
        if torch.any(torch.isinf(selected_frames)):
            raise ValueError("Inf detected in selected_frames.")

        ret = [selected_frames, selection_mask]
        if not self.training:
            ret.append(x_logits)
        return ret


class DTP(nn.Module):
    def __init__(self, config: DTPConfig, visual_hidden_size):
        super().__init__()
        self.config = config
        self.dtp_selector = SelectorModel(self.config)
        self.feature_extractor = VideoFeatureExtractor()
        self.visual_hidden_size = visual_hidden_size
        self.clip_to_image = nn.Linear(512, self.visual_hidden_size)

    def forward(self, image, t, frames_length=10, prompt="", N=0):
        frames = image[:, :, t:t + frames_length, :, :]
        if Discriminator(frames, frames_length, 0.9) == 1:
            print(prompt)
            with torch.no_grad():
                frames_embeds, prompt_embeds = self.feature_extractor(
                    video_frames=frames,
                    text_inputs=prompt
                )
            selected_frame_embeds, selection_mask = self.dtp_selector(  
                x_vis_seq=frames_embeds,
                x_txt_query=prompt_embeds,
                x_txt_cands=None
            )
            selected_frame_embeds = selected_frame_embeds.unsqueeze(1).expand(-1, N, -1)
            image_embeds = self.clip_to_image(selected_frame_embeds)
            return 1, image_embeds
        else:
            return 0, None
