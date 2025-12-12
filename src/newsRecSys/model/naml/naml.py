import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import BaseLightningModel từ file trước (hoặc BaseModel)
from newsRecSys.model.naml.base_model import BaseLightningModel
# Import các layer custom đã chuyển đổi ở bước 1
from newsRecSys.model.naml.layers import AttLayer2, SelfAttention

__all__ = ["NAMLModel"]


class NAMLModel(BaseLightningModel):
    """
    NAML model implementation in PyTorch.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        # Khởi tạo embedding trước khi gọi super().__init__
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.entity2vec_embedding = self._init_embedding(hparams.entityEmb_file)

        super(NAMLModel, self).__init__(hparams, iterator_creator, seed=seed)

    def _build_model(self):
        """Khởi tạo toàn bộ các layers của model."""
        hparams = self.hparams

        # 1. Embedding Layers
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.word2vec_embedding), freeze=False
        )
        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(self.entity2vec_embedding), freeze=False
        )
        self.vert_embedding = nn.Embedding(hparams.vert_num, hparams.vert_emb_dim)
        self.subvert_embedding = nn.Embedding(hparams.subvert_num, hparams.subvert_emb_dim)

        # 2. Sub-Encoders (CNN + Attention)
        # Title Encoder
        self.title_cnn = nn.Conv1d(
            in_channels=hparams.word_emb_dim,
            out_channels=hparams.filter_num,
            kernel_size=hparams.window_size,
            padding=hparams.window_size // 2  # Same padding logic
        )
        self.title_att = AttLayer2(hparams.filter_num, hparams.attention_hidden_dim, seed=self.seed)

        # Entity Title Encoder
        self.entity_title_cnn = nn.Conv1d(
            in_channels=hparams.entity_emb_dim,
            out_channels=hparams.filter_num,
            kernel_size=hparams.window_size,
            padding=hparams.window_size // 2
        )
        self.entity_title_att = AttLayer2(hparams.filter_num, hparams.attention_hidden_dim, seed=self.seed)

        # Projection cho Title (Title Rep + Entity Rep -> Dense)
        self.title_dense = nn.Linear(hparams.filter_num * 2, hparams.attention_hidden_dim)

        # Body Encoder
        self.body_cnn = nn.Conv1d(
            in_channels=hparams.word_emb_dim,
            out_channels=hparams.filter_num,
            kernel_size=hparams.window_size,
            padding=hparams.window_size // 2
        )
        self.body_att = AttLayer2(hparams.filter_num, hparams.attention_hidden_dim, seed=self.seed)

        # Entity Body Encoder
        self.entity_body_cnn = nn.Conv1d(
            in_channels=hparams.entity_emb_dim,
            out_channels=hparams.filter_num,
            kernel_size=hparams.window_size,
            padding=hparams.window_size // 2
        )
        self.entity_body_att = AttLayer2(hparams.filter_num, hparams.attention_hidden_dim, seed=self.seed)

        # Projection cho Body
        self.body_dense = nn.Linear(hparams.filter_num * 2, hparams.attention_hidden_dim)

        # Vert & Subvert Encoders
        self.vert_dense = nn.Linear(hparams.vert_emb_dim, hparams.filter_num)
        self.subvert_dense = nn.Linear(hparams.subvert_emb_dim, hparams.filter_num)

        # 3. Final News Attention (Aggregating Views)
        # Input dim = attention_hidden_dim (Title) + attention_hidden_dim (Body) + filter_num (Vert) + filter_num (Subvert)
        # Lưu ý: Trong code Keras gốc, các view được dense về attention_hidden_dim hoặc filter_num trước khi concat.
        # Ở đây tôi follow logic Keras: concat các view lại rồi qua AttLayer2.
        # Check logic code cũ: Title->Dense(att_dim), Body->Dense(att_dim), Vert->Dense(filter_num), Subvert->Dense(filter_num)
        # => Input cho Final Att là: att_dim + att_dim + filter_num + filter_num
        self.final_news_att = AttLayer2(
            hparams.attention_hidden_dim * 2 + hparams.filter_num * 2,
            hparams.attention_hidden_dim,
            seed=self.seed
        )

        # 4. User Encoder Layers
        # Input dim của SelfAttention là output dim của NewsEncoder
        # Output của NewsEncoder là sum weighted của input views -> dimension phụ thuộc vào input của AttLayer2
        # AttLayer2 trả về vector cùng chiều với input feature (theo axis 1).
        # Tuy nhiên, AttLayer2 trong code cũ trả về (Batch, Dim).
        # Logic Keras: Concat -> AttLayer2. AttLayer2 tính weight alpha, rồi sum(alpha * inputs).
        # Nhưng inputs ở đây là [TitleVec, BodyVec...] có chiều khác nhau?
        # A, logic Keras dùng `AttLayer2` với input shape (Batch, Views, Dim).
        # Code Keras: `concate_repr` axis=-2 (tạo sequence of views).
        # Điều này yêu cầu các View phải có cùng Dimension!
        # Code Keras: Title->Reshape(1, filter_num), Vert->Reshape(1, filter_num).
        # => Mọi view đều được project về `filter_num` (hoặc `attention_hidden_dim` nếu config set giống nhau).
        # Để an toàn và đúng logic Keras nhất: Ta giả định `attention_hidden_dim` == `filter_num`.

        self.news_dim = hparams.filter_num  # Theo output shape của các view encoder

        # Cần thêm layer projection nếu dim khác nhau (trong code Keras dùng Dense để ép về cùng dim trước khi concat)
        # Title: Dense(att_hidden_dim) -> Reshape(1, filter_num) => Output dim là filter_num.

        self.user_self_att = SelfAttention(
            input_dims=[self.news_dim, self.news_dim, self.news_dim],
            multiheads=hparams.head_num,
            head_dim=hparams.head_dim,
            seed=self.seed
        )
        self.user_att = AttLayer2(hparams.head_num * hparams.head_dim, hparams.attention_hidden_dim, seed=self.seed)

        self.dropout = nn.Dropout(hparams.dropout)

    def _get_input_label_from_iter(self, batch_data):
        """Chuyển đổi batch dictionary thành list các tensors."""
        # Gom các feature lại thành list hoặc dict để dễ truyền vào forward
        # Shape của mỗi tensor: (Batch, Seq_Len + 1, Feature_Len) (cho training)
        input_feat = {
            "title": torch.tensor(batch_data["candidate_title_batch"]),
            "entity_title": torch.tensor(batch_data["candidate_entity_title_batch"]),
            "body": torch.tensor(batch_data["candidate_ab_batch"]),
            "entity_body": torch.tensor(batch_data["candidate_entity_ab_batch"]),
            "vert": torch.tensor(batch_data["candidate_vert_batch"]),
            "subvert": torch.tensor(batch_data["candidate_subvert_batch"]),

            "his_title": torch.tensor(batch_data["clicked_title_batch"]),
            "his_entity_title": torch.tensor(batch_data["clicked_entity_title_batch"]),
            "his_body": torch.tensor(batch_data["clicked_ab_batch"]),
            "his_entity_body": torch.tensor(batch_data["clicked_entity_ab_batch"]),
            "his_vert": torch.tensor(batch_data["clicked_vert_batch"]),
            "his_subvert": torch.tensor(batch_data["clicked_subvert_batch"]),
        }
        input_label = torch.tensor(batch_data["labels"])
        return input_feat, input_label

    def _cnn_encoder(self, inputs, embedding_layer, cnn_layer, att_layer):
        """Helper cho việc: Emb -> Drop -> CNN -> Drop -> Att"""
        # inputs: (Batch, Seq_Len)
        x = embedding_layer(inputs)  # (B, L, D)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (B, D, L) cho Conv1d
        x = F.relu(cnn_layer(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (B, L, D) trả lại cho Attention
        x = att_layer(x)  # (B, D)
        return x

    def news_encoder(self, title, entity_title, body, entity_body, vert, subvert):
        """
        Mã hóa một tin tức (hoặc một batch các tin tức).
        Inputs shapes: (Batch, Seq_Len) cho text/entity, (Batch, 1) cho vert.
        """
        # 1. Title View
        t_vec = self._cnn_encoder(title, self.word_embedding, self.title_cnn, self.title_att)
        et_vec = self._cnn_encoder(entity_title, self.entity_embedding, self.entity_title_cnn, self.entity_title_att)
        # Concat & Project
        title_repr = torch.cat([t_vec, et_vec], dim=-1)  # (B, filter_num * 2)
        title_repr = self.title_dense(title_repr)  # (B, att_hidden_dim)
        # Reshape về (B, 1, filter_num) như logic Keras để chuẩn bị cho multi-view attention
        # Lưu ý: code Keras reshape title_repr (sau dense) thành (1, filter_num).
        # Điều này ngụ ý att_hidden_dim phải bằng filter_num * 1?
        # Giả sử hparams cấu hình hợp lệ, ta reshape theo config.
        title_repr = title_repr.view(-1, 1, self.hparams.filter_num)

        # 2. Body View
        b_vec = self._cnn_encoder(body, self.word_embedding, self.body_cnn, self.body_att)
        eb_vec = self._cnn_encoder(entity_body, self.entity_embedding, self.entity_body_cnn, self.entity_body_att)
        body_repr = torch.cat([b_vec, eb_vec], dim=-1)
        body_repr = self.body_dense(body_repr)
        body_repr = body_repr.view(-1, 1, self.hparams.filter_num)

        # 3. Category Views
        v_vec = self.vert_embedding(vert.squeeze(-1))  # (B, emb_dim)
        v_repr = F.relu(self.vert_dense(v_vec)).view(-1, 1, self.hparams.filter_num)

        sv_vec = self.subvert_embedding(subvert.squeeze(-1))
        sv_repr = F.relu(self.subvert_dense(sv_vec)).view(-1, 1, self.hparams.filter_num)

        # 4. Multi-view Attention
        # Stack views: (B, 4, filter_num)
        views = torch.cat([title_repr, body_repr, v_repr, sv_repr], dim=1)
        news_vec = self.final_news_att(views)  # (B, filter_num)

        return news_vec

    def user_encoder(self, his_feats):
        """
        Mã hóa người dùng từ lịch sử tin tức.
        his_feats: Dict chứa các tensor (Batch, His_Seq_Len, Feature_Dim)
        """
        # Batch size và History size
        batch_size = his_feats["his_title"].size(0)
        his_size = his_feats["his_title"].size(1)

        # 1. Reshape để đưa vào News Encoder (TimeDistributed simulation)
        # Gom (Batch, His) thành (Batch * His)
        flat_feats = {}
        for k, v in his_feats.items():
            # v: (B, His, ...)
            flat_feats[k.replace("his_", "")] = v.view(batch_size * his_size, -1)

        # 2. Encode từng tin trong lịch sử
        clicked_news_vecs = self.news_encoder(
            flat_feats["title"], flat_feats["entity_title"],
            flat_feats["body"], flat_feats["entity_body"],
            flat_feats["vert"], flat_feats["subvert"]
        )  # (B * His, News_Dim)

        # 3. Reshape lại thành (Batch, His, News_Dim)
        clicked_news_vecs = clicked_news_vecs.view(batch_size, his_size, -1)

        # 4. Self Attention
        # (Batch, His, Dim)
        y = self.user_self_att(clicked_news_vecs, clicked_news_vecs, clicked_news_vecs)

        # 5. Additive Attention để lấy User Vector
        user_vec = self.user_att(y)  # (Batch, Dim)
        return user_vec

    def forward(self, inputs):
        """
        Forward pass cho training/inference.
        inputs: Dict chứa tensors
        """
        # 1. Tính User Vector
        user_vec = self.user_encoder(inputs)  # (B, Dim)

        # 2. Tính Candidate News Vectors
        # inputs candidate shape: (B, 1 + npratio, Feature_Dim)
        batch_size = inputs["title"].size(0)
        cand_seq_len = inputs["title"].size(1)

        flat_cand = {}
        keys = ["title", "entity_title", "body", "entity_body", "vert", "subvert"]
        for k in keys:
            flat_cand[k] = inputs[k].view(batch_size * cand_seq_len, -1)

        cand_vecs = self.news_encoder(
            flat_cand["title"], flat_cand["entity_title"],
            flat_cand["body"], flat_cand["entity_body"],
            flat_cand["vert"], flat_cand["subvert"]
        )  # (B * Cand_Seq, Dim)

        cand_vecs = cand_vecs.view(batch_size, cand_seq_len, -1)  # (B, Cand_Seq, Dim)

        # 3. Dot Product (Interaction)
        # User: (B, Dim) -> (B, Dim, 1)
        # Cand: (B, Cand_Seq, Dim)
        # Score = Cand x User
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (B, Cand_Seq)

        return scores