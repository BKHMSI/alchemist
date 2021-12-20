import torch
import torch.nn as nn
import torch.nn.functional as F

from components.pos_encoding import PositionalEncoding
from components.multiheadattn import MultiHeadAttention

class A2C_EPN(nn.Module):
    def __init__(self, config, n_actions):
        super().__init__()

        self.obs_dim = config["obs-dim"]

        ############ A2C LSTM ############
        self.encoder = nn.Sequential( 
            nn.Linear(config["obs-dim"], config["encoder"][0]),
            nn.ELU(),
            nn.Linear(config["encoder"][0], config["encoder"][1]),
            nn.ELU(),
        ) 

        n_rewards = 1
        input_dim = config["encoder"][1] + n_actions + n_rewards + config['attn-dim']
        self.hidden_dim = config["hidden-dim"]
        self.lstm = nn.LSTMCell(input_dim, self.hidden_dim)

        self.actor = nn.Linear(self.hidden_dim, n_actions)
        self.critic = nn.Linear(self.hidden_dim, 1)

        ############ Multihead Attention ############
        n_feats, n_potions = 5, 8
        self.mem_size = config["dict-len"]
        self.attn_dim = config["attn-dim"]
        self.n_iter = config["attn-num-iter"]
        self.num_heads = config["attn-num-heads"]
        self.pos_enc = PositionalEncoding(self.attn_dim)
        self.attn_proj = nn.Linear(1+n_feats*2+n_potions+1+self.obs_dim, self.attn_dim)

        self.layer_norm = nn.LayerNorm((self.mem_size, self.attn_dim))
        self.attention = MultiHeadAttention(
            dim=self.attn_dim,
            q_dim=self.attn_dim,
            k_dim=self.attn_dim,
            v_dim=self.attn_dim,
            head_num=self.num_heads,
        )

        self.shared_mlp = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim),
            nn.ELU(),
            nn.Linear(self.attn_dim, self.attn_dim),
            nn.ELU(),
        )

    def forward(self, inputs):
 
        s_t, a_tm1, r_tm1, m_t, mask_t, lstm_state = inputs 

        s_t_expand = s_t.unsqueeze(1).expand(*m_t.shape[:-1], self.obs_dim)

        x = torch.cat([m_t, s_t_expand], dim=-1)
        b_i = self.pos_enc(self.attn_proj(x))
     
        x = self.layer_norm(b_i)
        attn_out, _ = self.attention(x, x, x, mask=mask_t.unsqueeze(1))
        b_i = F.elu(b_i + attn_out)

        b_i = self.shared_mlp(b_i)
        b_i = torch.max(b_i, dim=1).values

        feats = self.encoder(s_t) 
        x_t = torch.cat((feats, a_tm1, r_tm1, b_i), dim=-1)

        h_t, c_t = self.lstm(x_t, lstm_state)

        action_logits = self.actor(h_t)
        value_estimate = self.critic(h_t)

        return action_logits, value_estimate, (h_t, c_t)