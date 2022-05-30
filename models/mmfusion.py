# -*- coding: utf-8 -*-
# file: mmtan.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.mm_attention import MMAttention
from layers.attention2 import Attention2
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMFUSION(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MMFUSION, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                       batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                  batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                  batch_first=True)  # , dropout = opt.dropout_rate

        self.lstm_kernel1_ltext = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                              batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_kernel1_rtext = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                              batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_kernel1_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                               batch_first=True)  # , dropout = opt.dropout_rate

        self.lstm_kernel2_ltext = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                              batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_kernel2_rtext = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                              batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_kernel2_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                               batch_first=True)  # , dropout = opt.dropout_rate

        self.lstm_kernel3_ltext = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                              batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_kernel3_rtext = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                              batch_first=True)  # , dropout = opt.dropout_rate
        self.lstm_kernel3_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
                                               batch_first=True)  # , dropout = opt.dropout_rate

        self.attention_l = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.attention_r = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)

        self.attention_l2 = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.attention_r2 = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)

        self.attention_l3 = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.attention_r3 = Attention2(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)

        self.visaspect_att_l = MMAttention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.visaspect_att_r = MMAttention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        self.ltext2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.laspect2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.rtext2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.raspect2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        # self.viscontext_att_aspect = MMAttention(opt.hidden_dim, score_function='mlp', dropout=opt.dropout_rate)
        # self.visaspect_att_context = MMAttention(opt.hidden_dim, score_function='mlp', dropout=opt.dropout_rate)

        self.aspect2text = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.vismap2text = nn.Linear(2048, opt.hidden_dim)

        self.vis2text = nn.Linear(2048, opt.hidden_dim)
        self.vis2text_2 = nn.Linear(2048, opt.hidden_dim)
        self.vis2text_3 = nn.Linear(2048, opt.hidden_dim)

        self.gate = nn.Linear(2048 + 4 * opt.hidden_dim, opt.hidden_dim)

        self.madality_attetion = nn.Linear(opt.hidden_dim, 1)

        # blinear interaction between text vectors and image vectors
        # self.text2hidden = nn.Linear(opt.hidden_dim*3, opt.hidden_dim)
        # self.vis2hidden = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        # self.hidden2final = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        # self.text2hiddentext = nn.Linear(opt.hidden_dim*4, opt.hidden_dim*4)
        # self.vis2hiddentext = nn.Linear(opt.hidden_dim, opt.hidden_dim*4)

        self.text2hiddenvis = nn.Linear(opt.hidden_dim * 4, opt.hidden_dim)
        self.vis2hiddenvis = nn.Linear(opt.hidden_dim, opt.hidden_dim)

        # self.dense_2 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        self.dense_3 = nn.Linear(opt.hidden_dim*3, 10)
        # self.dense_4 = nn.Linear(opt.hidden_dim*4, opt.polarities_dim)
        # self.dense_5 = nn.Linear(opt.hidden_dim*5, opt.polarities_dim)
        # self.dense_10 = nn.Linear(opt.hidden_dim*10, opt.polarities_dim)
        if self.opt.att_mode == 'vis_concat_attimg' or self.opt.att_mode == 'vis_concat':
            self.dense_5 = nn.Linear(opt.hidden_dim * 5, opt.polarities_dim)
        elif self.opt.att_mode == 'vis_concat_attimg_gate':
            if self.opt.tfn:
                self.dense_6 = nn.Linear(opt.hidden_dim * opt.hidden_dim + 10, opt.polarities_dim)
            else:
                self.dense_6 = nn.Linear(opt.hidden_dim * 6 + 10, opt.polarities_dim)
        # self.dense_7 = nn.Linear(opt.hidden_dim*7, opt.polarities_dim)
        # self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def attention_linear(self, text, converted_vis_embed_map, vis_embed_map):

        """
        # 更改后的代码：
        # text: batch_size, hidden_dim; converted_vis_embed_map: batch_size, keys_number,embed_size; vis_embed_map:
        # batch_size, keys_number, 2048
        keys_size = converted_vis_embed_map.size(1)
        text = text.unsqueeze(1).expand(-1, keys_size, -1)  # batch_size, keys_number,hidden_dim

        # print("text.size()", text.size())
        # print("converted_vis_embed_map.size()", converted_vis_embed_map.size())

        if text.size() == converted_vis_embed_map.size():
            attention_inputs = torch.tanh(text + converted_vis_embed_map)
            # attention_inputs = F.dropout( attention_inputs )
            att_weights = self.madality_attetion(attention_inputs).squeeze(2)  # batch_size, keys_number
            att_weights = F.softmax(att_weights, dim=-1).view(-1, 1, 49)  # batch_size, 1, keys_number
            att_vector = torch.bmm(att_weights, vis_embed_map).view(-1, 2048)  # batch_size, 2048

        else:
            att_weights = self.madality_attetion(converted_vis_embed_map).squeeze(2)  # batch_size, keys_number
            att_weights = F.softmax(att_weights, dim=-1).view(-1, 1, 49)  # batch_size, 1, keys_number
            att_vector = torch.bmm(att_weights, vis_embed_map).view(-1, 2048)  # batch_size, 2048
        """

        keys_size = converted_vis_embed_map.size(1)
        text = text.unsqueeze(1).expand(-1, keys_size, -1)  # batch_size, keys_number,hidden_dim
        attention_inputs = torch.tanh(text + converted_vis_embed_map)
        # attention_inputs = F.dropout( attention_inputs )
        att_weights = self.madality_attetion(attention_inputs).squeeze(2)  # batch_size, keys_number
        att_weights = F.softmax(att_weights, dim=-1).view(-1, 1, 49)  # batch_size, 1, keys_number

        att_vector = torch.bmm(att_weights, vis_embed_map).view(-1, 2048)  # batch_size, 2048

        return att_vector, att_weights

    def forward(self, inputs, visual_embeds_global, visual_embeds_mean, visual_embeds_att,
                visual_embeds_global2, visual_embeds_mean2, visual_embeds_att2,
                visual_embeds_global3, visual_embeds_mean3, visual_embeds_att3, att_mode):

        x_l, x_r, aspect_indices = inputs[0], inputs[1], inputs[2]
        ori_x_l_len = torch.sum(x_l != 0, dim=-1)
        ori_x_r_len = torch.sum(x_r != 0, dim=-1)
        ori_aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        aspect = self.embed(aspect_indices)
        aspect_lstm, (_, _) = self.lstm_aspect(aspect, ori_aspect_len)
        aspect_len = torch.tensor(ori_aspect_len, dtype=torch.float).to(self.opt.device)
        sum_aspect = torch.sum(aspect_lstm, dim=1)
        avg_aspect = torch.div(sum_aspect, aspect_len.view(aspect_len.size(0), 1))

        # obtain the lstm hidden states for the left context and the right context respectively
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        # print("x_l.size():", x_l.size())        # torch.Size([10, 24, 300])
        # print("ori_x_l_len", ori_x_l_len)       # tensor([ 7,  5,  1,  7,  9, 14,  3,  4, 18,  8], device='cuda:0')
        l_context, (_, _) = self.lstm_l(x_l, ori_x_l_len)
        r_context, (_, _) = self.lstm_r(x_r, ori_x_r_len)

        # 原始代码就这一句话
        # converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        kernel1_out = torch.zeros([10, 10]).to(device)
        kernel2_out = torch.zeros([10, 10]).to(device)
        kernel3_out = torch.zeros([10, 10]).to(device)

        if visual_embeds_global.size() != torch.Size([0]):
            converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))
            # print(converted_vis_embed.size())     # torch.Size([10, 100])
            kernel1_ltext, (_, _) = self.lstm_kernel1_ltext(x_l, ori_x_l_len)
            kernel1_rtext, (_, _) = self.lstm_kernel1_rtext(x_r, ori_x_r_len)

            kernel1_aspect, (_, _) = self.lstm_kernel1_aspect(aspect, ori_aspect_len)
            kernel1_sum_aspect = torch.sum(kernel1_aspect, dim=1)
            kernel1_avg_aspect = torch.div(kernel1_sum_aspect, aspect_len.view(aspect_len.size(0), 1))

            kernel1_l_mid, kernel1_l_att = self.attention_l(kernel1_ltext, kernel1_avg_aspect, ori_x_l_len)
            kernel1_r_mid, kernel1_r_att = self.attention_r(kernel1_rtext, kernel1_avg_aspect, ori_x_r_len)
            kernel1_l_final = kernel1_l_mid.squeeze(dim=1)
            kernel1_r_final = kernel1_r_mid.squeeze(dim=1)

            kernel1_context = torch.cat((kernel1_l_final, kernel1_r_final), dim=-1)
            kernel1_x = torch.cat((kernel1_context, kernel1_avg_aspect), dim=-1)
            kernel1_out = self.dense_3(kernel1_x)

            # print("kernel1_out.size()", kernel1_out.size())     # torch.Size([10, 10])

        # print(visual_embeds_global.size())        # torch.Size([10, 2048])
        # print(visual_embeds_global2.size())       # torch.Size([0])

        if visual_embeds_global2.size() != torch.Size([0]):
            converted_vis_embed2 = self.vis2text_2(torch.tanh(visual_embeds_global2))
            # print(converted_vis_embed.size())
            kernel2_ltext, (_, _) = self.lstm_kernel2_ltext(x_l, ori_x_l_len)
            kernel2_rtext, (_, _) = self.lstm_kernel2_rtext(x_r, ori_x_r_len)

            kernel2_aspect, (_, _) = self.lstm_kernel2_aspect(aspect, ori_aspect_len)
            kernel2_sum_aspect = torch.sum(kernel2_aspect, dim=1)
            kernel2_avg_aspect = torch.div(kernel2_sum_aspect, aspect_len.view(aspect_len.size(0), 1))

            kernel2_l_mid, kernel2_l_att = self.attention_l2(kernel2_ltext, kernel2_avg_aspect, ori_x_l_len)
            kernel2_r_mid, kernel2_r_att = self.attention_r2(kernel2_rtext, kernel2_avg_aspect, ori_x_r_len)
            kernel2_l_final = kernel2_l_mid.squeeze(dim=1)
            kernel2_r_final = kernel2_r_mid.squeeze(dim=1)

            kernel2_context = torch.cat((kernel2_l_final, kernel2_r_final), dim=-1)
            kernel2_x = torch.cat((kernel2_context, kernel2_avg_aspect), dim=-1)
            kernel2_out = self.dense_3(kernel2_x)

            # print("kernel2_out.size()", kernel2_out.size())

        if visual_embeds_global3.size() != torch.Size([0]):
            converted_vis_embed3 = self.vis2text_3(torch.tanh(visual_embeds_global3))

            kernel3_ltext, (_, _) = self.lstm_kernel3_ltext(x_l, ori_x_l_len)
            kernel3_rtext, (_, _) = self.lstm_kernel3_rtext(x_r, ori_x_r_len)

            kernel3_aspect, (_, _) = self.lstm_kernel3_aspect(aspect, ori_aspect_len)
            kernel3_sum_aspect = torch.sum(kernel3_aspect, dim=1)
            kernel3_avg_aspect = torch.div(kernel3_sum_aspect, aspect_len.view(aspect_len.size(0), 1))

            kernel3_l_mid, kernel3_l_att = self.attention_l3(kernel3_ltext, kernel3_avg_aspect, ori_x_l_len)
            kernel3_r_mid, kernel3_r_att = self.attention_r3(kernel3_rtext, kernel3_avg_aspect, ori_x_r_len)
            kernel3_l_final = kernel3_l_mid.squeeze(dim=1)
            kernel3_r_final = kernel3_r_mid.squeeze(dim=1)

            kernel3_context = torch.cat((kernel3_l_final, kernel3_r_final), dim=-1)
            kernel3_x = torch.cat((kernel3_context, kernel3_avg_aspect), dim=-1)
            kernel3_out = self.dense_3(kernel3_x)

            # print("kernel3_out.size()", kernel3_out.size())

        kernel_out = (kernel1_out + kernel2_out + kernel3_out) / 3
        #
        # print(kernel_out.size())
        #
        if att_mode == 'text':  # apply aspect words to attend the left and right contexts
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)

            context = torch.cat((l_final, r_final), dim=-1)
            x = torch.cat((context, avg_aspect), dim=-1)
            out = self.dense_3(x)
            return out
        elif att_mode == 'vis_only':  # only use image and aspect words
            vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
            converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
            converted_aspect = self.aspect2text(avg_aspect)

            # att_vector: batch_size, 2048
            att_vector, att_weights = self.attention_linear(converted_aspect, converted_vis_embed_map, vis_embed_map)
            converted_att_vis_embed = self.vis2text(torch.tanh(att_vector))
            x = torch.cat((avg_aspect, converted_att_vis_embed), dim=-1)
            # x = torch.cat((avg_aspect, converted_vis_embed), dim=-1)
            out = self.dense_2(x)
            return out
        # elif att_mode == 'vis_concat':  # "text" mode concatenated with image
        #     l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
        #     r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
        #     l_final = l_mid.squeeze(dim=1)
        #     r_final = r_mid.squeeze(dim=1)
        #
        #     # """
        #     # low-rank pooling
        #     l_text = torch.tanh(self.ltext2hidden(l_final))  # batch_size, hidde_dim
        #     l_aspect = torch.tanh(self.laspect2hidden(avg_aspect))  # batch_size, hidden_dim
        #     ltext_aspect_inter = torch.mul(l_text, l_aspect)
        #     l_output = torch.cat((ltext_aspect_inter, l_final), dim=-1)
        #     # l_output = ltext_aspect_inter + l_final
        #
        #     r_text = torch.tanh(self.rtext2hidden(r_final))  # batch_size, hidde_dim
        #     r_aspect = torch.tanh(self.raspect2hidden(avg_aspect))  # batch_size, hidden_dim
        #     rtext_aspect_inter = torch.mul(r_text, r_aspect)
        #     r_output = torch.cat((rtext_aspect_inter, r_final), dim=-1)
        #     # r_output = rtext_aspect_inter + r_final
        #
        #     text_representation = torch.cat((l_output, r_output), dim=-1)
        #
        #     x = torch.cat((text_representation, converted_vis_embed), dim=-1)
        #
        #     x = self.dropout(x)
        #
        #     out = self.dense_5(x)
        #     return out
        elif att_mode == 'vis_concat_attimg':  # "text" mode concatenated with attention-based image
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)

            # """
            # low-rank pooling
            l_text = torch.tanh(self.ltext2hidden(l_final))  # batch_size, hidde_dim
            l_aspect = torch.tanh(self.laspect2hidden(avg_aspect))  # batch_size, hidden_dim
            ltext_aspect_inter = torch.mul(l_text, l_aspect)
            l_output = torch.cat((ltext_aspect_inter, l_final), dim=-1)
            # l_output = ltext_aspect_inter + l_final

            r_text = torch.tanh(self.rtext2hidden(r_final))  # batch_size, hidde_dim
            r_aspect = torch.tanh(self.raspect2hidden(avg_aspect))  # batch_size, hidden_dim
            rtext_aspect_inter = torch.mul(r_text, r_aspect)
            r_output = torch.cat((rtext_aspect_inter, r_final), dim=-1)
            # r_output = rtext_aspect_inter + r_final

            text_representation = torch.cat((l_output, r_output), dim=-1)

            vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
            converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
            converted_aspect = self.aspect2text(avg_aspect)

            # att_vector: batch_size, 2048
            att_vector, att_weights = self.attention_linear(converted_aspect, converted_vis_embed_map, vis_embed_map)
            converted_att_vis_embed = self.vis2text(torch.tanh(att_vector))

            x = torch.cat((text_representation, converted_att_vis_embed), dim=-1)
            x = self.dropout(x)

            out = self.dense_5(x)
            return out
        elif att_mode == 'vis_concat_attimg_gate':  # "text" mode concatenated with gated attention-based
            # image，首先采取此种方式！！！
            l_mid, l_att = self.attention_l(l_context, avg_aspect, ori_x_l_len)
            r_mid, r_att = self.attention_r(r_context, avg_aspect, ori_x_r_len)
            l_final = l_mid.squeeze(dim=1)
            r_final = r_mid.squeeze(dim=1)

            # context = torch.cat((l_final, r_final), dim=-1)
            # text_representation = torch.cat((context, avg_aspect), dim=-1)

            # """
            # low-rank pooling
            l_text = torch.tanh(self.ltext2hidden(l_final))  # batch_size, hidde_dim
            l_aspect = torch.tanh(self.laspect2hidden(avg_aspect))  # batch_size, hidden_dim
            ltext_aspect_inter = torch.mul(l_text, l_aspect)
            l_output = torch.cat((ltext_aspect_inter, l_final), dim=-1)
            # l_output = ltext_aspect_inter + l_final

            r_text = torch.tanh(self.rtext2hidden(r_final))  # batch_size, hidde_dim
            r_aspect = torch.tanh(self.raspect2hidden(avg_aspect))  # batch_size, hidden_dim
            rtext_aspect_inter = torch.mul(r_text, r_aspect)
            r_output = torch.cat((rtext_aspect_inter, r_final), dim=-1)
            # r_output = rtext_aspect_inter + r_final

            text_representation = torch.cat((l_output, r_output), dim=-1)
            # text_representation = torch.cat((text_representation, avg_aspect), dim=-1)
            # text_representation = self.dropout(text_representation)
            # context = torch.cat((l_output, r_output), dim=-1)
            # text_representation = torch.cat((context, avg_aspect), dim=-1)
            # """

            # apply entity-based attention mechanism to obtain different image representations
            vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
            converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, embed
            converted_aspect = self.aspect2text(avg_aspect)

            # att_vector: batch_size, 2048
            att_vector, att_weights = self.attention_linear(converted_aspect, converted_vis_embed_map, vis_embed_map)
            converted_att_vis_embed = self.vis2text(torch.tanh(att_vector))  # att_vector: batch_size, hidden_dim

            # print("text_representation.size()", text_representation.size())
            # print("att_vector.size()", att_vector.size())

            merge_representation = torch.cat((text_representation, att_vector), dim=-1)
            gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, hidden_dim
            gated_converted_att_vis_embed = torch.mul(gate_value, converted_att_vis_embed)
            # gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)

            # """
            # low-rank pooling
            # text_text = torch.tanh(self.text2hiddentext(text_representation))  # batch_size, hidde_dim
            # vis_text = torch.tanh(self.vis2hiddentext(gated_converted_att_vis_embed))  # batch_size, hidden_dim
            # vis_text_inter = torch.mul(text_text, vis_text)
            # text_output = torch.cat((text_representation, vis_text_inter), dim =-1)
            # text_output = text_representation + vis_text_inter
            # """

            # """
            text_vis = torch.tanh(self.text2hiddenvis(text_representation))  # batch_size, hidde_dim
            vis_vis = torch.tanh(self.vis2hiddenvis(gated_converted_att_vis_embed))  # batch_size, hidden_dim

            if self.opt.tfn:
                dot_matrix = torch.bmm(text_vis.unsqueeze(2), vis_vis.unsqueeze(1))
                x = dot_matrix.view(-1, self.opt.hidden_dim * self.opt.hidden_dim)
                x = self.dropout(x)
                out = self.dense_6(x)
            else:
                text_vis_inter = torch.mul(text_vis, vis_vis)
                vis_output = torch.cat((gated_converted_att_vis_embed, text_vis_inter, kernel_out), dim=-1)

                # comb = torch.cat((text_representation, gated_converted_att_vis_embed), dim=-1)
                # x = torch.cat((comb, text_vis_inter), dim=-1)
                x = torch.cat((text_representation, vis_output), dim=-1)
                x = self.dropout(x)
                out = self.dense_6(x)
            # """

            """
            without Multimodal Fusion (MF)
            vis_output = gated_converted_att_vis_embed
            #vis_output = gated_converted_att_vis_embed + text_vis_inter
            x = torch.cat((text_representation, vis_output), dim=-1)
            x = self.dropout(x)
            out = self.dense_5(x)
            """

            return out, l_att, r_att, att_weights

        # elif att_mode == 'vis_att':  # apply aspect words and image to attend the left and right contexts
        #     l_mid, _ = self.visaspect_att_l(l_context, avg_aspect, converted_vis_embed, \
        #                                     ori_x_l_len)
        #     r_mid, _ = self.visaspect_att_r(r_context, avg_aspect, converted_vis_embed, \
        #                                     ori_x_r_len)
        #     l_final = l_mid.squeeze(dim=1)
        #     r_final = r_mid.squeeze(dim=1)
        #
        #     context = torch.cat((l_final, r_final), dim=-1)
        #     x = torch.cat((context, avg_aspect), dim=-1)
        #
        #     out = self.dense_3(x)
        #     return out
