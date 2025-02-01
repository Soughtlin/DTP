from lavis.models.blip2_models.dtp_utils import *
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


def Discriminator(frames: torch.Tensor, segment_length: int, threshold: float) -> torch.Tensor:
    """
    计算每段相邻帧的相似度分数，并根据阈值返回1或0。
    Args:
        memory_bank (torch.Tensor): 输入记忆库。形状：(B, T, N, C)
        segment_length (int): 每段的帧数。
        threshold (float): 相似度阈值，范围在 (0, 1)。

    Returns:
        torch.Tensor: 每段的相似度判断结果。形状：(B, T-segment_length+1, N)，值为0或1。
    """
    B, T, N, C = frames.shape
    # 确保 segment_length 不大于 T
    assert segment_length <= T, "Segment length must be less than or equal to the total number of frames T."

    # 初始化结果张量
    result = torch.zeros((B, T - segment_length + 1, N), device=frames.device)

    # 遍历每一段
    for start in range(T - segment_length + 1):
        # 提取当前段
        segment = frames[:, start:start + segment_length, :, :]

        # 计算相邻帧之间的余弦相似度
        similarity_matrix = F.cosine_similarity(segment[:, :-1, :], segment[:, 1:, :], dim=-1)

        # 计算每段的平均相似度
        segment_similarity = similarity_matrix.mean(dim=1)  # 形状：(B, N)

        # 归一化相似度分数到 [0, 1]
        normalized_similarity = (segment_similarity - segment_similarity.min()) / (
                segment_similarity.max() - segment_similarity.min())

        # 根据阈值判断
        result[:, start, :] = (normalized_similarity > threshold).float()

    return result


class SelectorModel(nn.Module):
    """
    ATP选择模型。输入图像-文本编码序列，输出对输入帧的（离散）选择，帮助分析下游的图像-文本任务。
    """

    def __init__(self, config: DTPConfig, **kwargs):
        """
        config: ATPConfig，用于初始化ATPSelectorModel（及其编码器）的参数。
        具体请查看ATPConfig文档。
        """
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.d_input, config.d_model)  # 输入到模型的线性变换层
        self.atp_encoder = DTPEncoder(config, **kwargs)  # ATP编码器
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


class DTP(nn.Module):
    def __init__(self, config: DTPConfig, visual_hidden_size):
        super().__init__()
        self.config = config
        self.atp_selector = SelectorModel(self.config)
        self.feature_extractor = VideoFeatureExtractor()
        self.visual_hidden_size = visual_hidden_size
        self.clip_to_image = nn.Linear(512, self.visual_hidden_size)

    def forward(self, image, t, frames_length=10, prompt="", N=0):
        # print(f"ori_image_embeds: {image_embeds.shape}")
        frames = image[:, :, t:t + frames_length, :, :]
        if Discriminator(frames, frames_length, 0.9) == 1:
            print(prompt)
            # print(f"frames input: {frames.shape}")
            with torch.no_grad():  # [B,frame_step, D]
                frames_embeds, prompt_embeds = self.feature_extractor(
                    video_frames=frames,
                    text_inputs=prompt
                )
            # print(f"frames embeds: {frames_embeds.shape}")
            # print(f"prompt embeds: {prompt_embeds.shape}")
            selected_frame_embeds, selection_mask = self.atp_selector(  # 输出的图像[B,D]
                x_vis_seq=frames_embeds,
                x_txt_query=prompt_embeds,
                x_txt_cands=None
            )
            selected_frame_embeds = selected_frame_embeds.unsqueeze(1).expand(-1, N, -1)
            # 将512映射到要求的维度上，这个或者也可以驾到atp前面
            image_embeds = self.clip_to_image(selected_frame_embeds)
            # print(f"select image embeds: {image_embeds.shape}")
            return 1, image_embeds
        else:
            return 0, None
