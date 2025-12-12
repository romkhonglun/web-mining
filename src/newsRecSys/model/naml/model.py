import torch
import torch.nn as nn


# -----------------------------
# Additive Attention
# -----------------------------
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # x: (B, T, H)
        score = torch.tanh(self.proj(x)) @ self.v  # (B, T)
        attn = torch.softmax(score, dim=1).unsqueeze(-1)  # (B, T, 1)
        out = (x * attn).sum(dim=1)                     # (B, H)
        return out

# -----------------------------
# Multi-Head Self Attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


# -----------------------------
# Encoder for Title, subtitle, etc.
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.additive = AdditiveAttention(embed_dim)

    def forward(self, x):
        # x: (B, T)
        x = self.embedding(x)
        x = self.self_attn(x)              # (B, T, H)
        x = self.additive(x)               # (B, H)
        return x


# -----------------------------
# News Encoder (NAML core)
# -----------------------------
class NewsEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        num_heads=20,
        category_size=100,
        category_dim=100
    ):
        super().__init__()

        # Multi-view encoders
        self.title_encoder = TextEncoder(vocab_size, embed_dim, num_heads)
        self.subtitle_encoder = TextEncoder(vocab_size, embed_dim, num_heads)

        # Category embeddings
        self.category_embedding = nn.Embedding(category_size, category_dim)
        self.subcategory_embedding = nn.Embedding(category_size, category_dim)

        # Projection to final dim
        self.final_dim = embed_dim * 2 + category_dim * 2
        self.proj = nn.Linear(self.final_dim, embed_dim)

    def forward(self, title, subtitle, category, subcategory):
        # text views
        title_vec = self.title_encoder(title)
        abs_vec = self.subtitle_encoder(subtitle)

        # category views
        cat_vec = self.category_embedding(category)
        subcat_vec = self.subcategory_embedding(subcategory)

        # concat all views
        x = torch.cat([title_vec, abs_vec, cat_vec, subcat_vec], dim=-1)

        # project
        out = self.proj(x)
        return out   # (B, embed_dim)


# -----------------------------
# User Encoder
# -----------------------------
class UserEncoder(nn.Module):
    def __init__(self, embed_dim=300, num_heads=20):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.additive = AdditiveAttention(embed_dim)

    def forward(self, clicked_news_vecs):
        # clicked_news_vecs: (B, N, H)
        x = self.self_attn(clicked_news_vecs)
        x = self.additive(x)  # (B, H)
        return x


# -----------------------------
# NAML Model
# -----------------------------
class NAML(nn.Module):
    def __init__(
        self,
        vocab_size,
        category_size,
        embed_dim=300,
        num_heads=20
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            category_size=category_size,
            category_dim=embed_dim // 3,
        )
        self.user_encoder = UserEncoder(embed_dim, num_heads)

    def forward(
        self,
        candidate_news,
        clicked_news
    ):
        """
        candidate_news:
            title, subtitle, category, subcategory
            Each: (B, T)
        clicked_news:
            encoded news vectors already or raw inputs
        """

        # Encode candidate
        c_title, c_abs, c_cat, c_subcat = candidate_news
        cand_vec = self.news_encoder(c_title, c_abs, c_cat, c_subcat)

        # Encode clicked history
        h_titles, h_abs, h_cat, h_subcat = clicked_news  # each: (B, N, T)
        B, N, T = h_titles.shape

        # reshape to (B*N, T)
        h_title = h_titles.reshape(B * N, T)
        h_abs = h_abs.reshape(B * N, T)
        h_cat = h_cat.reshape(B * N)
        h_sub = h_subcat.reshape(B * N)

        # encode each clicked article
        h_vec = self.news_encoder(h_title, h_abs, h_cat, h_sub)
        h_vec = h_vec.reshape(B, N, -1)

        # user vector
        user_vec = self.user_encoder(h_vec)

        # score via dot product
        score = (cand_vec * user_vec).sum(dim=-1)  # (B,)
        return score
