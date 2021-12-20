import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, softmax_dim, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=softmax_dim)
        return attention.matmul(value), attention


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 dim,
                 q_dim,
                 k_dim,
                 v_dim,
                 head_num,
                 bias=True,
                 activation=F.relu,
                 softmax_dim=-1):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if dim % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(dim, head_num))
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.dim = dim
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.softmax_dim = softmax_dim
        self.linear_q = nn.Linear(self.q_dim, dim, bias)
        self.linear_k = nn.Linear(self.k_dim, dim, bias)
        self.linear_v = nn.Linear(self.v_dim, dim, bias)
        self.linear_o = nn.Linear(dim, dim, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn_weights = ScaledDotProductAttention()(q, k, v, self.softmax_dim, mask)
        y = self._reshape_from_batches(y)
        attn_weights = self._reshape_from_batches(attn_weights)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn_weights

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)-torch.eye(seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.dim, self.head_num, self.bias, self.activation,
        )