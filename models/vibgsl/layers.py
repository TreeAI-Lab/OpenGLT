import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


VERY_SMALL_NUMBER = 1e-12
INF = 1e20


def normalize_adj(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum.clamp(min=VERY_SMALL_NUMBER), -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, graph_type, top_k=None,
                 epsilon=None, num_pers=4, metric_type='attention',
                 feature_denoise=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_pers = num_pers
        self.graph_type = graph_type
        self.top_k = top_k
        self.epsilon = epsilon
        self.metric_type = metric_type
        self.feature_denoise = feature_denoise

        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([
                nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)
            ])
        elif metric_type == 'weighted_cosine':
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(num_pers, input_size)))
        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([
                nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)
            ])
            self.linear_sims2 = nn.ModuleList([
                nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)
            ])
            self.leakyrelu = nn.LeakyReLU(0.2)
        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)
        elif metric_type == 'cosine':
            pass
        elif metric_type == 'mlp':
            self.lin1 = nn.Linear(input_size, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
        elif metric_type == 'multi_mlp':
            self.linear_sims1 = nn.ModuleList([
                nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)
            ])
            self.linear_sims2 = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_pers)
            ])
        else:
            raise ValueError(f'Unknown metric_type: {metric_type}')

        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(input_size, init_strategy='constant')

    def reset_parameters(self):
        if self.feature_denoise:
            self.feat_mask = self.construct_feat_mask(self.input_size, init_strategy='constant')
        if self.metric_type == 'attention':
            for module in self.linear_sims:
                module.reset_parameters()
        elif self.metric_type == 'weighted_cosine':
            nn.init.xavier_uniform_(self.weight_tensor)
        elif self.metric_type == 'gat_attention':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()
        elif self.metric_type == 'kernel':
            self.precision_inv_dis.data.uniform_(0, 1.0)
            nn.init.xavier_uniform_(self.weight)
        elif self.metric_type == 'transformer':
            self.linear_sim1.reset_parameters()
            self.linear_sim2.reset_parameters()
        elif self.metric_type == 'mlp':
            self.lin1.reset_parameters()
            self.lin2.reset_parameters()
        elif self.metric_type == 'multi_mlp':
            for module in self.linear_sims1:
                module.reset_parameters()
            for module in self.linear_sims2:
                module.reset_parameters()

    def forward(self, node_features):
        if self.feature_denoise:
            masked_features = self.mask_feature(node_features)
            learned_adj = self.learn_adj(masked_features)
            return masked_features, learned_adj
        learned_adj = self.learn_adj(node_features)
        return node_features, learned_adj

    def learn_adj(self, context, ctx_mask=None):
        if self.metric_type == 'attention':
            attention = 0
            for linear in self.linear_sims:
                context_fc = torch.relu(linear(context))
                attention = attention + torch.matmul(context_fc, context_fc.transpose(-1, -2))
            attention = attention / len(self.linear_sims)
            markoff_value = -INF
        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0
        elif self.metric_type == 'transformer':
            q = self.linear_sim1(context)
            attention = torch.matmul(q, q.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            markoff_value = -INF
        elif self.metric_type == 'gat_attention':
            attention = []
            for lin1, lin2 in zip(self.linear_sims1, self.linear_sims2):
                attention.append(self.leakyrelu(lin1(context) + lin2(context).transpose(-1, -2)))
            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF
        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis ** 2))
            markoff_value = 0
        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True).clamp(min=VERY_SMALL_NUMBER))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0
        elif self.metric_type == 'mlp':
            context_fc = torch.relu(self.lin2(torch.relu(self.lin1(context))))
            attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))
            markoff_value = 0
        elif self.metric_type == 'multi_mlp':
            attention = 0
            for lin1, lin2 in zip(self.linear_sims1, self.linear_sims2):
                context_fc = torch.relu(lin2(torch.relu(lin1(context))))
                attention = attention + torch.matmul(context_fc, context_fc.transpose(-1, -2))
            attention = attention / self.num_pers
            markoff_value = -INF

        if ctx_mask is not None:
            attention = attention.masked_fill(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.graph_type == 'epsilonNN':
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        elif self.graph_type == 'KNN':
            attention = self.build_knn_neighbourhood(attention, self.top_k, markoff_value)
        elif self.graph_type == 'prob':
            attention = self.build_prob_neighbourhood(attention, temperature=0.05)
        else:
            raise ValueError(f'Unknown graph_type: {self.graph_type}')

        if self.graph_type in ['KNN', 'epsilonNN']:
            if self.metric_type in ('kernel', 'weighted_cosine'):
                attention = attention / torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            elif self.metric_type == 'cosine':
                attention = (attention > 0).float()
                attention = normalize_adj(attention)
            elif self.metric_type in ('transformer', 'attention', 'gat_attention'):
                attention = torch.softmax(attention, dim=-1)
        return attention

    @staticmethod
    def build_knn_neighbourhood(attention, top_k, markoff_value):
        top_k = min(int(top_k), attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, top_k, dim=-1)
        return (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)

    @staticmethod
    def build_epsilon_neighbourhood(attention, epsilon, markoff_value):
        attention = torch.sigmoid(attention)
        mask = (attention > float(epsilon)).detach().float()
        return attention * mask + markoff_value * (1 - mask)

    @staticmethod
    def build_prob_neighbourhood(attention, temperature=0.1):
        attention = torch.clamp(attention, 0.01, 0.99)
        weighted_adj = RelaxedBernoulli(
            temperature=torch.tensor([temperature], device=attention.device),
            probs=attention,
        ).rsample()
        mask = (weighted_adj > 0.5).detach().float()
        return weighted_adj * mask

    def compute_distance_mat(self, x, weight=None):
        trans_x = torch.mm(x, weight) if weight is not None else x
        norm = torch.sum(trans_x * x, dim=-1)
        return -2 * torch.matmul(trans_x, x.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)

    @staticmethod
    def construct_feat_mask(feat_dim, init_strategy='normal'):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        with torch.no_grad():
            if init_strategy == 'normal':
                mask.normal_(1.0, 0.1)
            elif init_strategy == 'constant':
                nn.init.constant_(mask, 0.0)
            else:
                raise ValueError(f'Unknown init_strategy: {init_strategy}')
        return mask

    def mask_feature(self, x, use_sigmoid=True, marginalize=True):
        feat_mask = torch.sigmoid(self.feat_mask) if use_sigmoid else self.feat_mask
        feat_mask = feat_mask.to(x.device)
        if marginalize:
            std_tensor = torch.ones_like(x, dtype=torch.float) / 2
            mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
            z = torch.normal(mean=mean_tensor, std=std_tensor).to(x.device)
            return x + z * (1 - feat_mask)
        return x * feat_mask
