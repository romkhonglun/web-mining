import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttLayer2(nn.Module):
    """Soft alignment attention implementation."""

    def __init__(self, input_dim, dim=200, seed=0):
        """
        Args:
            input_dim (int): dimension of input feature (last dimension of input).
            dim (int): attention hidden dim.
        """
        super(AttLayer2, self).__init__()
        self.dim = dim
        self.seed = seed

        # Thiết lập seed (nếu cần thiết cho reproducibility)
        torch.manual_seed(seed)

        # W: transformation weight (tương đương W và b trong code cũ)
        # Input: (batch, seq_len, input_dim) -> Output: (batch, seq_len, dim)
        self.W = nn.Linear(input_dim, dim, bias=True)

        # q: context vector
        # Shape: (dim, 1)
        self.q = nn.Parameter(torch.Tensor(dim, 1))

        # Khởi tạo weights giống Glorot Uniform (Xavier)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (Batch, Seq_Len, Input_Dim)
            mask: (Batch, Seq_Len) - Boolean or 0/1 mask
        """
        # 1. Tính toán attention score (phi tuyến)
        # u = tanh(W.x + b)
        u = torch.tanh(self.W(inputs))  # (B, L, dim)

        # 2. Tính độ tương đồng với context vector q
        # attention = u . q
        attention = torch.matmul(u, self.q).squeeze(-1)  # (B, L)

        # 3. Xử lý Masking và Normalization
        if mask is not None:
            # Chuyển mask sang float nếu cần
            if mask.dtype == torch.bool:
                mask = mask.float()

            # Trong TF code cũ: exp(att) * mask.
            # Cách ổn định hơn trong PyTorch: mask fill với giá trị cực nhỏ rồi softmax
            attention = attention.masked_fill(mask == 0, -1e9)

        # Softmax để lấy trọng số (alpha)
        attention_weights = F.softmax(attention, dim=-1)  # (B, L)

        # 4. Weighted Sum
        # Mở rộng chiều để nhân: (B, L, 1) * (B, L, D)
        weighted_input = inputs * attention_weights.unsqueeze(-1)

        # Sum theo chiều sequence (axis 1) -> (B, D)
        output = torch.sum(weighted_input, dim=1)

        return output


class SelfAttention(nn.Module):
    """Multi-head self attention implementation."""

    def __init__(self, input_dims, multiheads, head_dim, seed=0, mask_right=False):
        """
        Args:
            input_dims (list of int): [Q_dim, K_dim, V_dim] - số chiều của input features.
            multiheads (int): Số lượng heads.
            head_dim (int): Số chiều của mỗi head.
            mask_right (bool): Che phần tương lai (dùng cho decoder/causal).
        """
        super(SelfAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed

        torch.manual_seed(seed)

        # Q, K, V projections
        # Lưu ý: input_dims phải là list [dim_q, dim_k, dim_v]
        self.WQ = nn.Linear(input_dims[0], self.output_dim)
        self.WK = nn.Linear(input_dims[1], self.output_dim)
        self.WV = nn.Linear(input_dims[2], self.output_dim)

        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)

    def _create_sequence_mask(self, lengths, max_len):
        """Tạo mask từ chiều dài chuỗi (batch_size, 1)"""
        # lengths: (B, 1) hoặc (B,)
        lengths = lengths.view(-1)
        # Tạo range (0, 1, 2, ..., max_len-1)
        range_tensor = torch.arange(max_len, device=lengths.device)
        # So sánh để tạo mask (B, max_len)
        mask = range_tensor[None, :] < lengths[:, None]
        return mask.float()

    def forward(self, q_seq, k_seq, v_seq, q_len=None, v_len=None):
        """
        Args:
            q_seq, k_seq, v_seq: Tensors (Batch, Seq_Len, Dim)
            q_len, v_len: Tensors chiều dài thực tế (Batch, 1) hoặc None
        """
        batch_size = q_seq.size(0)

        # 1. Linear Projections
        Q = self.WQ(q_seq)  # (B, L_q, output_dim)
        K_proj = self.WK(k_seq)  # (B, L_k, output_dim)
        V = self.WV(v_seq)  # (B, L_v, output_dim)

        # 2. Reshape & Permute cho Multi-head
        # (B, L, heads, head_dim) -> (B, heads, L, head_dim)
        Q = Q.view(batch_size, -1, self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K_proj = K_proj.view(batch_size, -1, self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        # (B, heads, L_q, head_dim) * (B, heads, L_k, head_dim) -> (B, heads, L_q, L_k)
        # einsum equivalent: torch.matmul(Q, K.transpose(-1, -2))
        scores = torch.matmul(Q, K_proj.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        # 4. Masking
        # Mask padding (V_len dùng để mask Key/Value)
        if v_len is not None:
            # Tạo mask (B, L_k)
            seq_mask = self._create_sequence_mask(v_len, K_proj.size(2))
            # Reshape để khớp (B, 1, 1, L_k) để broadcast cho tất cả heads và query positions
            seq_mask = seq_mask.view(batch_size, 1, 1, -1)
            scores = scores.masked_fill(seq_mask == 0, -1e12)  # Use very small number like TF code

        # Mask Right (Causal Mask)
        if self.mask_right:
            # Tạo tam giác trên (upper triangular)
            L_q, L_k = Q.size(2), K_proj.size(2)
            ones = torch.ones((L_q, L_k), device=scores.device)
            causal_mask = torch.tril(ones)  # Giữ lại phần dưới
            scores = scores.masked_fill(causal_mask == 0, -1e12)

        # 5. Softmax & Output
        A = F.softmax(scores, dim=-1)  # (B, heads, L_q, L_k)

        # (B, heads, L_q, L_k) * (B, heads, L_v, head_dim) -> (B, heads, L_q, head_dim)
        O = torch.matmul(A, V)

        # Permute lại: (B, L_q, heads, head_dim)
        O = O.permute(0, 2, 1, 3).contiguous()

        # Flatten: (B, L_q, output_dim)
        O = O.view(batch_size, -1, self.output_dim)

        # Mask output padding (nếu Q_len được cung cấp, ta zero phần padding ở output)
        if q_len is not None:
            mask_out = self._create_sequence_mask(q_len, O.size(1))  # (B, L_q)
            mask_out = mask_out.unsqueeze(-1)  # (B, L_q, 1)
            O = O * mask_out  # Mode "mul" trong TF code

        return O


class PersonalizedAttentivePooling(nn.Module):
    """
    Soft alignment attention implementation.
    Trong PyTorch, đây là một Module chứ không phải hàm trả về Model.
    """

    def __init__(self, dim2, dim3, seed=0):
        """
        Args:
            dim2 (int): dimension of input vectors (input_dim).
            dim3 (int): dimension of query vector.
        """
        super(PersonalizedAttentivePooling, self).__init__()
        torch.manual_seed(seed)

        self.dropout = nn.Dropout(0.2)
        # Dense layer chiếu input vecs sang không gian query
        self.user_att_layer = nn.Linear(dim2, dim3, bias=True)
        nn.init.xavier_uniform_(self.user_att_layer.weight)
        nn.init.zeros_(self.user_att_layer.bias)

    def forward(self, vecs_input, query_input):
        """
        Args:
            vecs_input: (Batch, Seq_Len, dim2)
            query_input: (Batch, dim3)
        Returns:
            user_vec: (Batch, dim2)
        """
        # 1. Dropout
        user_vecs = self.dropout(vecs_input)  # (B, L, D2)

        # 2. Project vectors -> (B, L, D3)
        user_att = torch.tanh(self.user_att_layer(user_vecs))

        # 3. Calculate Attention Scores
        # Dot product giữa projected vectors và query
        # query_input: (B, D3) -> unsqueeze -> (B, D3, 1)
        # user_att: (B, L, D3)
        # bmm: (B, L, D3) x (B, D3, 1) -> (B, L, 1)
        scores = torch.bmm(user_att, query_input.unsqueeze(-1))

        # 4. Softmax
        att_weights = F.softmax(scores, dim=1)  # (B, L, 1)

        # 5. Weighted Sum
        # (B, D2, L) x (B, L, 1) -> (B, D2, 1) -> (B, D2)
        # Transpose vecs_input to (B, D2, L) for easy multiplication
        user_vec = torch.bmm(vecs_input.transpose(1, 2), att_weights).squeeze(-1)

        return user_vec


class ComputeMasking(nn.Module):
    """Compute if inputs contains zero value."""

    def __init__(self):
        super(ComputeMasking, self).__init__()

    def forward(self, inputs):
        # inputs != 0 -> True -> Float (1.0)
        return (inputs != 0).float()


class OverwriteMasking(nn.Module):
    """Set values at specific positions to zero."""

    def __init__(self):
        super(OverwriteMasking, self).__init__()

    def forward(self, values, mask):
        """
        Args:
            values: Tensor (B, ...)
            mask: Tensor (B, ...) cùng shape hoặc broadcast được
        """
        # Nếu mask thiếu chiều cuối (ví dụ mask shape (B, L) nhưng value (B, L, D))
        if values.dim() == 3 and mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        return values * mask