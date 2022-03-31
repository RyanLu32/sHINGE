import torch, math, itertools, os, psutil
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class HINGE(torch.nn.Module):

    def __init__(self, n_rel_keys, num_values, num_types, embedding_size, num_filters=200):
        super(HINGE, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters

        self.f_FCN_net = torch.nn.Linear((num_filters * (embedding_size - 2)) * 2, 1)  # type * 2
        xavier_normal_(self.f_FCN_net.weight.data)  # initialize weight
        zeros_(self.f_FCN_net.bias.data)  # initialize bias

        # initialize role embedding
        self.emb_relations_keys = torch.nn.Embedding(n_rel_keys, self.embedding_size, padding_idx=0)
        # initialize value embedding
        self.emb_entities_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # initialize type embedding
        self.emb_types = torch.nn.Embedding(num_types, self.embedding_size, padding_idx=0)

        # conv for hrt
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3))  # input_channel = 1; kernal size = 3*3
        zeros_(self.conv1.bias.data)  # initialize conv1 bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)  # channel_num = num_filters
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)  # initialize conv1 weight

        # conv for hrtkv
        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, 3))
        zeros_(self.conv2.bias.data)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1)

        # conv for hrt_type
        self.conv3 = torch.nn.Conv2d(1, num_filters, (3, 3))
        zeros_(self.conv3.bias.data)
        self.batchNorm3 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv3.weight, mean=0.0, std=0.1)

        # conv for hrtkv_type
        self.conv4 = torch.nn.Conv2d(1, num_filters, (5, 3))
        zeros_(self.conv4.bias.data)
        self.batchNorm4 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv4.weight, mean=0.0, std=0.1)

        bound = math.sqrt(1.0 / self.embedding_size)
        uniform_(self.emb_relations_keys.weight.data, -bound, bound)  # further initialize the role embedding
        uniform_(self.emb_entities_values.weight.data, -bound, bound)
        uniform_(self.emb_types.weight.data, -bound, bound)

        self.loss = torch.nn.Softplus()

    def forward(self, x_batch, arity, num_tuple, device=None):

        # entity branch
        fact_rel_keys_ids = torch.LongTensor(
            np.array(x_batch[:, 0::2][:, 0:2]).flatten()).cuda(device)  # [[rel1,rel1],[rel2,rel2]]
        fact_head_tail_values_ids = torch.LongTensor(
            np.array(x_batch[:, 1::2][:, 0:2]).flatten()).cuda(device)  # [[head1,tail1],[head2,tail2]]
        fact_rel_keys_embedded = self.emb_relations_keys(fact_rel_keys_ids).view(len(x_batch), 2,
                                                                                 self.embedding_size)  # fact_rel_keys_embedded also contains the relation_tail which is not used by HINGE because it is the same as the relation_head. however, we kept it to be consistent with the NaLP datastructure since the pre-processing scripts are the same.
        fact_head_tail_values_embedded = self.emb_entities_values(fact_head_tail_values_ids).view(len(x_batch), 2,
                                                                                                  self.embedding_size)
        if arity > 2:
            kv_keys_ids = torch.LongTensor(np.array(x_batch[:, 0::2][:, 2:arity]).flatten()).cuda(device)
            kv_values_ids = torch.LongTensor(np.array(x_batch[:, 1::2][:, 2:arity]).flatten()).cuda(device)
            kv_keys_embedded = self.emb_relations_keys(kv_keys_ids).view(len(x_batch), arity - 2, self.embedding_size)
            kv_values_embedded = self.emb_entities_values(kv_values_ids).view(len(x_batch), arity - 2,
                                                                              self.embedding_size)

        fact_hrt_concat1 = torch.cat((fact_head_tail_values_embedded[:, 0, :].unsqueeze(1),
                                      fact_rel_keys_embedded[:, 0, :].unsqueeze(1),
                                      fact_head_tail_values_embedded[:, 1, :].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1)
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)

        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)


        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_head_tail_values_embedded[:, 0, :].unsqueeze(1),
                                              fact_rel_keys_embedded[:, 0, :].unsqueeze(1),
                                              fact_head_tail_values_embedded[:, 1, :].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_keys_embedded[:, 0, :].unsqueeze(1),
                                              kv_values_embedded[:, 0, :].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp)
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0),
                                                                             -1).unsqueeze(2)
            fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)

            for i in range(arity - 3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_keys_embedded[:, i + 1, :].unsqueeze(1),
                                                  kv_values_embedded[:, i + 1, :].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp)
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0),
                                                                                 -1).unsqueeze(2)
                fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
            
        
        min_val_entity, _ = torch.min(fact_hrt_concat_vectors1, 2)

        
        
        # type branch
        start_point = arity * 2
        head_types_ids = torch.LongTensor(np.array(x_batch[:, start_point::arity]).flatten()).cuda(device)
        tail_types_ids = torch.LongTensor(np.array(x_batch[:, (start_point + 1)::arity]).flatten()).cuda(device)
        head_types_embedded = self.emb_types(head_types_ids).view(len(x_batch), num_tuple - 2, self.embedding_size)
        tail_types_embedded = self.emb_types(tail_types_ids).view(len(x_batch), num_tuple - 2, self.embedding_size)

        headType_relation_tailType_concat = torch.cat((head_types_embedded[:, 0, :].unsqueeze(1),
                                                       fact_rel_keys_embedded[:, 0, :].unsqueeze(1),
                                                       tail_types_embedded[:, 0, :].unsqueeze(1)), 1).unsqueeze(1)
        headType_relation_tailType_concat = self.conv3(headType_relation_tailType_concat)
        headType_relation_tailType_concat = self.batchNorm3(headType_relation_tailType_concat)
        headType_relation_tailType_concat = F.relu(headType_relation_tailType_concat).squeeze(3)
        headType_relation_tailType_concat = headType_relation_tailType_concat.view(
            headType_relation_tailType_concat.size(0), -1).unsqueeze(2)

        for i in range(num_tuple - 3):
            headType_relation_tailType_concat_tmp = torch.cat((head_types_embedded[:, i + 1, :].unsqueeze(1),
                                                               fact_rel_keys_embedded[:, 0, :].unsqueeze(1),
                                                               tail_types_embedded[:, i + 1, :].unsqueeze(1)),
                                                              1).unsqueeze(1)
            headType_relation_tailType_concat_tmp = self.conv3(headType_relation_tailType_concat_tmp)
            headType_relation_tailType_concat_tmp = self.batchNorm3(headType_relation_tailType_concat_tmp)
            headType_relation_tailType_concat_tmp = F.relu(headType_relation_tailType_concat_tmp).squeeze(3)
            headType_relation_tailType_concat_tmp = headType_relation_tailType_concat_tmp.view(
                headType_relation_tailType_concat_tmp.size(0), -1).unsqueeze(2)
            headType_relation_tailType_concat = torch.cat(
                (headType_relation_tailType_concat, headType_relation_tailType_concat_tmp), 2)

        if arity > 2:
            for i in range(arity - 2):
                value_types_ids = torch.LongTensor(np.array(x_batch[:, (start_point + i + 2)::arity]).flatten()).cuda(device)
                value_types_embedded = self.emb_types(value_types_ids).view(len(x_batch), num_tuple - 2,
                                                                            self.embedding_size)

                headType_relation_tailType_key_valueType_concat_iter = torch.cat((head_types_embedded[:, 0, :].unsqueeze(1),
                                                                             fact_rel_keys_embedded[:, 0, :].unsqueeze(1),
                                                                             tail_types_embedded[:, 0, :].unsqueeze(1),
                                                                             kv_keys_embedded[:, i, :].unsqueeze(1),
                                                                             value_types_embedded[:, 0, :].unsqueeze(1)), 1).unsqueeze(1)
                headType_relation_tailType_key_valueType_concat_iter = self.conv4(headType_relation_tailType_key_valueType_concat_iter)
                headType_relation_tailType_key_valueType_concat_iter = self.batchNorm4(headType_relation_tailType_key_valueType_concat_iter)
                headType_relation_tailType_key_valueType_concat_iter = F.relu(headType_relation_tailType_key_valueType_concat_iter).squeeze(3)
                headType_relation_tailType_key_valueType_concat_iter = headType_relation_tailType_key_valueType_concat_iter.view(
                    headType_relation_tailType_key_valueType_concat_iter.size(0), -1).unsqueeze(2)

                for j in range(num_tuple - 3):
                    headType_relation_tailType_key_valueType_concat_tmp = torch.cat((head_types_embedded[:, j + 1, :].unsqueeze(1),
                                                                             fact_rel_keys_embedded[:, 0, :].unsqueeze(1),
                                                                             tail_types_embedded[:, j + 1, :].unsqueeze(1),
                                                                             kv_keys_embedded[:, i, :].unsqueeze(1),
                                                                             value_types_embedded[:, j + 1, :].unsqueeze(1)), 1).unsqueeze(1)
                    headType_relation_tailType_key_valueType_concat_tmp = self.conv4(
                        headType_relation_tailType_key_valueType_concat_tmp)
                    headType_relation_tailType_key_valueType_concat_tmp = self.batchNorm4(
                        headType_relation_tailType_key_valueType_concat_tmp)
                    headType_relation_tailType_key_valueType_concat_tmp = F.relu(
                        headType_relation_tailType_key_valueType_concat_tmp).squeeze(3)
                    headType_relation_tailType_key_valueType_concat_tmp = headType_relation_tailType_key_valueType_concat_tmp.view(
                        headType_relation_tailType_key_valueType_concat_tmp.size(0), -1).unsqueeze(2)
                    headType_relation_tailType_key_valueType_concat_iter = torch.cat(
                        (headType_relation_tailType_key_valueType_concat_iter, headType_relation_tailType_key_valueType_concat_tmp), 2)
                    
                    
                    
                headType_relation_tailType_concat = torch.cat((headType_relation_tailType_concat, headType_relation_tailType_key_valueType_concat_iter), 2)
                

                
        min_val_type, _ = torch.min(headType_relation_tailType_concat, 2)
             
                
        # concatenate entity and type
        concat_fact_and_type = torch.cat((min_val_entity, min_val_type), 1)
        evaluation_score = self.f_FCN_net(concat_fact_and_type)


        return evaluation_score
