"""
Main file containing core ATP class code and some example utilities using ATP.
"""

from lavis.models.blip2_models.atp_utils import ModalityEmbeddings
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

import math

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
class ATPConfig:
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

class ATPEncoder(nn.Module):
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

class ATPSelectorModel(nn.Module):
    """
    ATP选择模型。输入图像-文本编码序列，输出对输入帧的（离散）选择，帮助分析下游的图像-文本任务。
    """

    def __init__(self, config: ATPConfig, **kwargs):
        """
        config: ATPConfig，用于初始化ATPSelectorModel（及其编码器）的参数。
        具体请查看ATPConfig文档。
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_input, config.d_model)  # 输入到模型的线性变换层
        self.atp_encoder = ATPEncoder(config, **kwargs)  # ATP编码器
        self.dropout = nn.Dropout(p=config.sel_dropout)  # dropout层，用于防止过拟合
        self.logits = nn.Linear(config.d_model, 1)  # 输出的logits层，用于生成选择分数

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_query: Optional[torch.tensor] = None,
                x_txt_cands: Optional[torch.tensor] = None,
                **kwargs):
        """
        执行ATP选择操作，输入为图像-文本编码序列，返回选择的（未修改的）视觉嵌入和选择掩码。
        x_vis_seq: 形状为(B, T, D_in)的视觉特征嵌入，其中：
            B: 批量大小 (batch_size)
            T: 序列长度 (frame_count)，即视频帧数
            D_in: 输入特征维度 (input_feature_dimension)，即每帧图像的特征数量

        x_txt_query: 形状为(B, D_in)的可选查询文本嵌入，其中：
            B: 批量大小 (batch_size)
            D_in: 查询文本的特征维度 (input_feature_dimension)

        x_txt_cands: 形状为(B, L_cands, D_in)的可选候选文本嵌入，其中：
            B: 批量大小 (batch_size)
            L_cands: 候选文本的数量 (number_of_candidates)
            D_in: 每个候选文本的特征维度 (input_feature_dimension)
        (可选) temperature: 当配置use_ste为False时使用的温度，默认为1.0。
        """
        B, T, D = x_vis_seq.size()  # 获取批量大小、视频帧数、特征维度
        x_vis_seq = x_vis_seq.permute(1, 0, 2)  # 将x_vis_seq变为(T, B, D)，使序列在前

        n_text_feats = self.atp_encoder.modality_encoding.n_text_feats  # 获取文本特征的数量

        # 将输入合并成一个多模态序列
        x_inputs = []
        if self.config.use_text_query:
            assert x_txt_query is not None, "缺少x_txt_query。"  # 确保提供了查询文本
            x_inputs.append(x_txt_query.unsqueeze(0))  # 将查询文本添加到输入序列
        if self.config.use_text_cands:
            assert x_txt_cands is not None, "缺少x_txt_cands。"  # 确保提供了候选文本
            x_inputs.append(x_txt_cands.permute(1, 0, 2))  # 将候选文本添加到输入序列
        x_inputs.append(x_vis_seq)  # 将视觉特征添加到输入序列
        x_inputs = torch.cat(x_inputs, dim=0)  # 将所有输入序列在第0维拼接

        # 对输入序列进行嵌入，映射到更小的维度（d_model），并加入模态编码
        x_encoded = self.embedding(self.dropout(x_inputs))  # 应用线性变换和dropout
        x_atp_encoded = self.atp_encoder(x_encoded)[n_text_feats:]  # 通过ATP编码器

        # 获取选择分数（logits）
        x_logits = self.logits(self.dropout(x_atp_encoded))  # 通过logits层生成选择分数

        # 检查x_logits是否存在NaN或Inf
        if torch.any(torch.isnan(x_logits)):
            raise ValueError("x_logits中检测到NaN！")
        if torch.any(torch.isinf(x_logits)):
            raise ValueError("x_logits中检测到Inf！")

        # 获取选择掩码
        if self.training:
            if self.config.use_ste:
                # Gumbel软最大化
                selection_mask = F.gumbel_softmax(x_logits, dim=0, hard=True)
            else:
                # 使用温度调节的软最大化
                temperature = kwargs.get("temperature", 1.0)
                if temperature <= 0:
                    raise ValueError(f"无效的温度：{temperature}")
                selection_mask = F.softmax(x_logits / temperature, dim=0)
        else:
            # 在评估时使用硬选择
            selection_index_argmax = x_logits.max(dim=0, keepdim=True)[1]  # 选择最大分数的索引
            selection_mask = torch.zeros_like(x_logits, memory_format=torch.contiguous_format).scatter_(
                dim=0, index=selection_index_argmax, value=1.0
            )  # 根据最大分数的索引设置选择掩码为1

        # 检查selection_mask是否存在NaN或Inf
        if torch.any(torch.isnan(selection_mask)):
            raise ValueError("selection_mask中检测到NaN！")
        if torch.any(torch.isinf(selection_mask)):
            raise ValueError("selection_mask中检测到Inf！")

        # 使用选择掩码执行帧选择
        selected_frames = (selection_mask * x_vis_seq).sum(dim=0)  # 根据掩码选择帧，得到(B, D_in)

        # 检查selected_frames是否存在异常值
        if torch.any(torch.isnan(selected_frames)):
            raise ValueError("selected_frames中检测到NaN！")
        if torch.any(torch.isinf(selected_frames)):
            raise ValueError("selected_frames中检测到Inf！")

        # 返回选择的帧和选择掩码
        ret = [selected_frames, selection_mask]
        if not self.training:
            ret.append(x_logits)
        return ret

# 实现一个反向筛选器
def select_frame_from_mask(frames, selection_mask):
    """
    Select a single frame from a batch of videos based on a one-hot selection mask.

    Args:
        frames (torch.Tensor): Input video frames, shape (B, C, T, H, W).
        selection_mask (torch.Tensor): One-hot selection mask, shape (T, B, 1).

    Returns:
        torch.Tensor: Selected frames, shape (B, C, 1, H, W).
    """
    # print(f"frames shape: {frames.shape}")
    # print(f"selection mask shape: {selection_mask.shape}")
    # Transpose selection_mask to match frames' batch dimension
    selection_mask = selection_mask.permute(1, 2, 0)  # Shape: (B, 1, T)

    # Find the indices where the mask is 1
    selected_indices = torch.argmax(selection_mask, dim=-1).squeeze(-1)  # Shape: (B)
    # print(f"selected indices shape: {selected_indices.shape}")

    # Gather the selected frame
    B, C, T, H, W = frames.shape
    selected_frames = frames[torch.arange(B), :, selected_indices, :, :]  # Shape: (B, C, H, W)
    # print(f"selected frames shape: {selected_frames.shape}")

    return selected_frames.unsqueeze(2)  # Shape:(B, C, 1, H, W)


######
# Below are some utility functions (and illustrative examples) for using ATP in the context of a 
# broader script, for visualization and inference. See also training_example.py.
def get_selected_index_from_selection_mask(frame_idxs_gt, selection_mask, sequence_first=False):
    """
    Quick utility helper method to get the "groundtruth" frame index selected
    (assuming shuffled input, and groundtruth frame indexes of (N, L) are available).
    This is useful for visualizations of ATP predictions on the original (ordered) video.
    """
    fidxs_gt = frame_idxs_gt.transpose(0, 1) if not sequence_first else frame_idxs_gt  # (L, N)
    print(f"selection_mask shape: {selection_mask.shape}")
    print(f"fidxs_gt shape: {fidxs_gt.shape}")
    print(selection_mask)
    return (selection_mask.squeeze(-1) * fidxs_gt).sum(dim=0)


def atp_downstream_task_forward(atp_selector: ATPSelectorModel, batch, **kwargs):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """
    x_vis_seq, frame_idxs_gt, x_txt_query, x_txt_cands, y_gt = batch
    # note: frame_idxs_gt only here for potential visualization; not used by ATP.
    selected_frames, *out_masks = atp_selector(x_vis_seq, x_txt_query, x_txt_cands, **kwargs)
    y_pred = F.cosine_similarity(selected_frames.unsqueeze(1), x_txt_cands, dim=-1)  # (N, N_ans)
    loss = F.cross_entropy(y_pred, y_gt)
    accs = (y_pred.argmax(dim=-1) == y_gt).float()
    return (loss, accs, selected_frames, y_pred, out_masks)


def apply_additional_masks(x, additional_masks=[]):
    """
    To enable combination of ATP with other (complementary) techniques, we provide this example (optional) function.
    Use to combine the outputs (masks) of these other methods with ATP. Does nothing if additional_masks is empty.
    Modify to fit your specific task use case.
    """
    x_out = x
    for mask in additional_masks:
        x_out *= mask
    return x_out


def atp_downstream_task_forward_with_additional_masks(atp_selector: ATPSelectorModel, batch, **kwargs):
    """
    Replica of atp_downstream_task_forward, with some examples for incorporating additional masks.
    Note: default behavior of this function is identical to atp_downstream_task_forward without 
    any additional masks. Modify to fit your specific task use case.
    """
    x_vis_seq, frame_idxs_gt, x_txt_query, x_txt_cands, y_gt, *additional_masks = batch
    # note: frame_idxs_gt only here for potential visualization; not used by ATP.
    if kwargs.get("apply_additional_masks_pre_atp", False):
        assert len(additional_masks) > 0, "additional_masks is empty, nothing to apply pre-ATP"
        x_txt_cands = apply_additional_masks(x_txt_cands, additional_masks)
    selected_frames, *out_masks = atp_selector(x_vis_seq, x_txt_query, x_txt_cands, **kwargs)

    y_pred = F.cosine_similarity(selected_frames.unsqueeze(1), x_txt_cands, dim=-1)  # (N, N_ans)
    if kwargs.get("apply_additional_masks_preds", False):
        assert len(additional_masks) > 0, "additional_masks is empty, nothing to apply post-ATP"
        y_pred = apply_additional_masks(y_pred, additional_masks)

    loss = F.cross_entropy(y_pred, y_gt)
    accs = (y_pred.argmax(dim=-1) == y_gt).float()
    return (loss, accs, selected_frames, y_pred, out_masks)

######
