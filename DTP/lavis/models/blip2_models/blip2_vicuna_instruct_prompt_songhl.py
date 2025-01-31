"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
import imageio.config.plugins
from packaging import version
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import transformers
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train, memory_bank_compress


# prompt + image -> cross attention
class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout=0.1):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # 多头注意力的线性层
        self.query_fc = nn.Linear(embed_size, hidden_size)
        self.key_fc = nn.Linear(embed_size, hidden_size)
        self.value_fc = nn.Linear(embed_size, hidden_size)
        # 最终输出的线性层
        self.out_fc = nn.Linear(hidden_size, embed_size)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, query, key, value, attention_mask=None):
        # query, key, value 的形状都是 [B, T, embed_size]
        B, T_q, _ = query.size()
        _, T_k, _ = key.size()
        # 线性变换
        Q = self.query_fc(query)  # [B, T_q, embed_size]
        K = self.key_fc(key)  # [B, T_k, embed_size]
        V = self.value_fc(value)  # [B, T_k, embed_size]
        # 拆分为多头 [B,T,h,D]
        # [B,Tokens,num_heads,head_dim]再对第一和第二维度转置
        assert self.hidden_size % self.num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        head_dim = self.hidden_size // self.num_heads
        Q = Q.view(B, T_q, self.num_heads, head_dim).transpose(1, 2)  # [B,h,T_q,D]
        K = K.view(B, T_k, self.num_heads, head_dim).transpose(1, 2)  # [B,h,T_k,D]
        V = V.view(B, T_k, self.num_heads, head_dim).transpose(1, 2)  # [B,h,T_k,D]
        # 计算点积注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B,h,T_q,D] * [B,h,D,T_k] -> Q[B, h, T_q, T_k]
        attention_scores = attention_scores / head_dim ** 0.5  # 缩放
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask  # 加入mask（可选）
        # Softmax 和 Dropout
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # [B, h, T_q, T_k]
        attention_weights = self.dropout(attention_weights)
        # 加权求和得到输出
        attention_output = torch.matmul(attention_weights, V)  # [B, h, T_q, D]
        # 合并多头输出# [B, T_q, D]
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)
        # attention经过线型层输出
        attention_output = self.out_fc(attention_output)  # [B, T_q, embed_size]

        return attention_output


@registry.register_model("blip2_vicuna_instruct_prompt_malmm")
class Blip2VicunaInstruct_MALMM(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            llm_model="",
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
            memory_bank_length=0,
            num_frames=0,
            max_num_frames=120,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, memory_bank_length=memory_bank_length,
            num_frames=num_frames,
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.num_query_token = num_query_token
        self.memory_bank_length = memory_bank_length
        self.use_memory_bank = True if memory_bank_length > 0 else False
        self.num_frames = num_frames
        self.visual_memory_bank = None
        self.image_pe = nn.Embedding(max_num_frames, 1408)
        nn.init.constant_(self.image_pe.weight, 0.0)

        # visual_encoder与bert(prompt_embedding)特征纬度匹配
        self.bert_hidden_size = self.Qformer.config.hidden_size  # BERT 的隐藏维度
        self.visual_hidden_size = self.visual_encoder.num_features  # visual_encoder的隐藏维度
        # 如果维度不一致，添加线性层映射
        if self.bert_hidden_size != self.visual_hidden_size:
            self.linear_text_to_visual = nn.Linear(self.bert_hidden_size, self.visual_hidden_size)
        else:
            self.linear_text_to_visual = None

        # 定义Cross Attention层,用于visual_encoder和prompt
        self.cross_attention = CrossAttention(
            embed_size=self.visual_hidden_size,
            num_heads=8,
            hidden_size=self.visual_hidden_size,
            dropout=0.1
        )

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        # For video data
        is_video = False
        if image.dim() == 5:  # 检查是否为视频数据
            is_video = True
            B, C, T, H, W = image.shape  # batchsize, channels, time(frame), height, width

        if self.qformer_text_input:  # 检查是否用QFormer处理文本输入
            if is_video:
                # 将输入文本 samples["text_input"] 按时间帧数 T 进行重复，以便与视频的每一帧对应
                text_input = [text for text in samples["text_input"] for _ in range(T)]
            else:
                text_input = samples["text_input"]

            if self.use_memory_bank:
                # 将[1,32,C]的query扩展到[B, 32, C],从而与输入的视频特征纬度(batchsize)匹配
                query_tokens = self.query_tokens.expand(B, -1, -1)
                text_Qformer = self.tokenizer(
                    samples["text_input"],  # [B]
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                # 生成[B, N]大小全为1(有效)的mask，[:-1]舍弃了最后一个纬度C(不需要表示特征维度的具体信息,只判定哪些有效那些无效)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                # 将查询 token和文本token的mask拼接，使得整个Q-Former可以同时处理查询和文本的注意力交互
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
                # 逐帧遍历
                # torch.autograd.set_detect_anomaly(True)
                for t in range(T):
                    with self.maybe_autocast():
                        # 抽取batch中每个sample的t帧->[B,C,H,W](batch中的每个样本是独立计算的，不会混淆)
                        # 经过视觉编码器后，C,H,W纬度消失，转变成patch(256+1,1可能类似CLS token用于全局信息聚合)
                        # 视觉编码之后的特征向量纬度的大小为1408(C)，ln_vision是一个归一化模块
                        image_embeds = self.ln_vision(self.visual_encoder(image[:, :, t, :, :]))  # [B, 256+1, 1408]
                    N, C = image_embeds.shape[-2:]

                    # Position Encoding,[1]的元素值就是标量t,代表视频的帧序号，即时间信息
                    position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                    position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                    # image_pe应该会把[B,1]扩展为[B,N,C]
                    image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]
                    image_embeds_ori = image_embeds  # [B,N,C]保留不含的image_embeds用于后面cross attention
                    # 给image扩展出时间维度
                    image_embeds = image_embeds.unsqueeze(1)  # [B, 1, N, C]
                    # image mask,用在提取视觉特征的cross-attention模块
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]

                    # 给prompt使用bert编码 [batchsize, sequence_length, C]
                    # batchsize能保持一致吗？ 可以
                    # 方案1
                    prompt_embeds = self.Qformer.bert.embeddings(text_Qformer.input_ids)
                    # 方案2
                    # prompt_embeds = query_output.last_hidden_state  # [B, T, D]

                    # !!! 将cross_attention放在bert前面，其输出
                    # 让 prompt 与 visual encoder 进行cross attention
                    if self.linear_text_to_visual is not None:
                        prompt_embeds = self.linear_text_to_visual(prompt_embeds)  # [B, T, C]
                    # 使用 Cross Attention，将 image_embeds 作为 Q， 作为prompt_embeds K 和 V
                    # 但我c个人认为将 prompt_embeds 作为 Q，image_embeds 作为 K 和 V更好，但不符合visual_encoder纬度要求

                    self.cross_attention = self.cross_attention.to(image.device)  # 这个to(device)应该可以放在下面
                    prompt_visual_cross = self.cross_attention(
                        query=image_embeds_ori,  # [B, N, C]
                        key=prompt_embeds,  # [B, T_text, C]
                        value=prompt_embeds,  # [B, T_text, C]
                        attention_mask=None
                    )  # 输出[B, N, C]

                    # 残差连接
                    # try1:将prompt与cross输出加和
                    prompt_visual_cross = image_embeds_ori + prompt_visual_cross  # [B, N, C]
                    # try2(个人认为更合理):将prompt与cross输出加和，想达到残差连接or保留了原始的文本信息的目的
                    # prompt_visual_cross = prompt_embeds + prompt_visual_cross  # [B, T_text, C]

                    # 将cross attention输出扩展时间帧纬度
                    prompt_visual_cross = prompt_visual_cross.unsqueeze(1)  # [B,1,N,C]

                    # 如果是第一帧，初始化visual_memory_bank作为第一帧的嵌入
                    if t == 0:
                        # !!! encoder_hidden_states 直接使用ross_attention的输出进行初始化
                        encoder_hidden_states = prompt_visual_cross  # [B, 1, N, C] encoder_hidden_states表示模型当前对视觉特征的理解
                        # compress(不可微?)之前要把梯度断掉，detach不能随便动
                        # 学习.detach() .clone()  .cpu()v .numpy()
                        self.visual_memory_bank = prompt_visual_cross.detach()  # [B, 1, N, C]
                        # size_constant会用于后续压缩
                        self.size_constant = torch.ones(B, 1, N).to(image_embeds.device)  # [B, 1, N]
                        self.compression_size = self.size_constant  # [B, 1, N]
                    else:
                        # !!! encoder_hidden_states 直接使用ross_attention的输出进行更新
                        encoder_hidden_states = torch.cat([self.visual_memory_bank, prompt_visual_cross],
                                                          dim=1)  # [B, (t+1), N, C]
                        # 将visual_memory_bank与当前帧嵌入连接，并更新compression_size
                        self.visual_memory_bank = torch.cat([self.visual_memory_bank, prompt_visual_cross.detach()],
                                                            dim=1)  # [B, t+1, N, C]
                        self.compression_size = torch.cat([self.compression_size, self.size_constant],
                                                          dim=1)  # [B, t+1, N]

                    # bert模型
                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=encoder_hidden_states.view(B, -1, C), #
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    # print("t = ",t) #调试

                    # 如果是最后一帧，删除visual_memory_bank和compression_size
                    # 否则，如果当前visual_memory_bank的长度超过阈值，则压缩visual_memory_bank
                    if t == T - 1:
                        # print("T = ",T)
                        del self.visual_memory_bank
                        del self.compression_size
                    elif self.visual_memory_bank.size(1) > self.memory_bank_length:
                        # print("Start compressing")
                        self.visual_memory_bank, self.compression_size = memory_bank_compress(
                            self.visual_memory_bank,
                            self.compression_size)
            else:  # 不使用memory bank
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                text_Qformer = self.tokenizer(
                    text_input,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                if is_video:
                    image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image))  # [B * T, 256+1, 1408]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:  # 无文本输入
            if is_video:
                image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))  # [B * T, 256+1, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(B * T, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        if is_video:
            inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        print("Prompt:", prompt)
        if isinstance(prompt, list):
            print("Prompt Lengths:", [len(p) for p in prompt])

        image = samples["image"]
        # For video data
        is_video = False
        if image.dim() == 5:
            is_video = True
            B, C, T, H, W = image.shape

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        assert self.qformer_text_input == True
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            if is_video:
                text_input = []
                for text in prompt:
                    text_input.extend([text] * T)
            else:
                text_input = prompt

            if self.use_memory_bank:
                query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 32, C]
                text_Qformer = self.tokenizer(
                    prompt,  # [B]
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)  # [B, 32]
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                for t in range(T):
                    with self.maybe_autocast():
                        image_embeds = self.ln_vision(
                            self.visual_encoder(image[:, :, t, :, :])).detach()  # [B, 256+1, 1408]
                    N, C = image_embeds.shape[-2:]

                    # Position Encoding,[1]的元素值就是标量t,代表视频的帧序号，即时间信息
                    position_ids = torch.tensor([t]).long().to(image_embeds.device)  # [1]
                    position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
                    # image_pe应该会把[B,1]扩展为[B,N,C]
                    image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]
                    image_embeds_ori = image_embeds  # [B,N,C]保留不含的image_embeds用于后面cross attention
                    # 给image扩展出时间维度
                    image_embeds = image_embeds.unsqueeze(1)  # [B, 1, N, C]
                    # image mask,用在提取视觉特征的cross-attention模块
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # [B, N]

                    # 给prompt使用bert编码
                    # [batchsize, sequence_length, C] batchsize能保持一致吗？
                    # 方案1
                    prompt_embeds = self.Qformer.bert.embeddings(text_Qformer.input_ids)
                    # 方案2
                    # prompt_embeds = query_output.last_hidden_state  # [B, T, D]
                    # 让 prompt 与 visual encoder 进行cross attention
                    if self.linear_text_to_visual is not None:
                        prompt_embeds = self.linear_text_to_visual(prompt_embeds)  # [B, T, C]
                    # 使用 Cross Attention，将 image_embeds 作为 Q， 作为prompt_embeds K 和 V
                    # 但我个人认为将 prompt_embeds 作为 Q，image_embeds 作为 K 和 V更好，但不符合visual_encoder纬度要求
                    self.cross_attention = self.cross_attention.to(image.device)
                    prompt_visual_cross = self.cross_attention(
                        query=image_embeds_ori,  # [B, N, C]
                        key=prompt_embeds,  # [B, T_text, C]
                        value=prompt_embeds,  # [B, T_text, C]
                        attention_mask=None
                    )  # 输出[B, N, C]

                    # 残差连接
                    # try1:将prompt与cross输出加和
                    prompt_visual_cross = image_embeds_ori + prompt_visual_cross  # [B, N, C]
                    # try2(个人认为更合理):将prompt与cross输出加和，想达到残差连接or保留了原始的文本信息的目的
                    # prompt_visual_cross = prompt_embeds + prompt_visual_cross  # [B, T_text, C]

                    # 将cross attention输出扩展时间帧纬度
                    prompt_visual_cross = prompt_visual_cross.unsqueeze(1)  # [B,1,N,C]

                    # If it is the first frame, initialize the visual_memory_bank as the embedding of the first frame
                    # If not, concatenate the visual_memory_bank with the current frame embedding and update the compression_size
                    if t == 0:
                        self.visual_memory_bank = prompt_visual_cross  # [B, 1, N, C]
                        self.size_constant = torch.ones(B, 1, N).to(image_embeds.device)  # [B, 1, N]
                        self.compression_size = self.size_constant  # [B, 1, N]
                    else:
                        self.visual_memory_bank = torch.cat([self.visual_memory_bank, prompt_visual_cross],
                                                            dim=1)  # [B, t+1, N, C]
                        self.compression_size = torch.cat([self.compression_size, self.size_constant],
                                                          dim=1)  # [B, t+1, N]

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=self.visual_memory_bank.view(B, -1, C),
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    if t == T - 1:
                        del self.visual_memory_bank
                        del self.compression_size
                    elif self.visual_memory_bank.size(1) > self.memory_bank_length:
                        self.visual_memory_bank, self.compression_size = memory_bank_compress(
                            self.visual_memory_bank,
                            self.compression_size)

            else:
                query_tokens = self.query_tokens.expand(B * T, -1, -1)
                text_Qformer = self.tokenizer(
                    text_input,  # [B*T]
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                if is_video:
                    image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image))  # [B * T, 256+1, 1408]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        else:
            if is_video:
                image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))  # [B * T, 256+1, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(B * T, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        if is_video:
            inputs_llm = inputs_llm.reshape(B, -1, inputs_llm.shape[-1])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,  # 这里有问题
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                max_new_tokens=max_length  # Here
            )

        outputs[outputs < 2] = 2  # convert output id -1/0/1 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in
                                        enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_captions=num_beams,
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in
                      range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id,
                                                              -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")
        memory_bank_length = cfg.get("memory_bank_length", 0)
        num_frames = cfg.get("num_frames", 0)
        max_num_frames = cfg.get("max_num_frames", 120)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            memory_bank_length=memory_bank_length,
            num_frames=num_frames,
            max_num_frames=max_num_frames,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model
