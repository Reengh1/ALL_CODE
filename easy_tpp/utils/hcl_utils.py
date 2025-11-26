import torch
import math
def event2seq_embedding(enc_out, non_pad_mask):
    """
    enc_out: event embedding with size batch x seq_len x d_model
    non_pad_mask: with size batch x seq_len x 1
    """
    seq_emb = torch.sum(enc_out * non_pad_mask, dim=1) / (torch.sum(non_pad_mask, dim=1) + 1e-6)  # batch * d_model
    return seq_emb


def event_contrastive_loss(all_lambda, types, non_pad_mask):
    num_types = all_lambda.shape[2]
    type_mask = torch.zeros([*types.size(), num_types], device=all_lambda.device)  # batch * seq_len * num_types
    for i in range(num_types):
        type_mask[:, :, i] = (types == i).bool().to(all_lambda.device)

    pos_event_lambda = torch.sum(all_lambda * type_mask, dim=2, keepdim=True)  # batch * seq_len * 1
    neg_event_lambda = all_lambda * (1 - type_mask)  # batch * seq_len * num_types
    all_event_lambda = torch.sum(all_lambda + 1e-8, dim=2, keepdim=True)  # batch * seq_len * 1

    pos_p = (pos_event_lambda + 1e-8) / (all_event_lambda + 1e-8)  # batch * seq_len * 1
    neg_p = 1 - neg_event_lambda / (all_event_lambda + 1e-8)  # batch * seq_len * num_types
    cl1 = -torch.mean(torch.log(pos_p + 1e-8) * non_pad_mask)
    cl2 = -torch.mean(torch.sum(torch.log(neg_p + 1e-8) * non_pad_mask, dim=2))
    return cl1 + cl2


def seq_contrastive_loss(seq_emb, pos_seq_emb, neg_seq_emb, scalar: float = 10):
    """
    Sequence level contrastive loss
    seq_emb: batch * d_model
    pos_seq_emb: batch * d_model
    neg_seq_emb: (batch * K) * d_model
    """
    batch = seq_emb.shape[0]
    num_neg = int(neg_seq_emb.shape[0] / seq_emb.shape[0])
    seq_emb = seq_emb / (torch.sqrt(torch.sum(seq_emb ** 2, dim =1, keepdim=True)) + 1e-8)
    pos_seq_emb = pos_seq_emb / (torch.sqrt(torch.sum(pos_seq_emb ** 2, dim=1, keepdim=True)) + 1e-8)
    neg_seq_emb = neg_seq_emb / (torch.sqrt(torch.sum(neg_seq_emb ** 2, dim=1, keepdim=True)) + 1e-8)
    pos_v = torch.exp(scalar * torch.sum(seq_emb * pos_seq_emb, dim=1, keepdim=True))  # batch x 1
    for k in range(num_neg):
        neg_seq_emb[k*batch:(k+1)*batch, :] = seq_emb * neg_seq_emb[k*batch:(k+1)*batch, :].clone()
    neg_v = torch.exp(scalar * torch.sum(neg_seq_emb, dim=1))  # (batch * num_neg)
    neg_v = torch.reshape(neg_v, (batch, num_neg))  # batch x num_neg

    all_v = torch.sum(neg_v, dim=1, keepdim=True) + pos_v + 1e-8  # batch x 1
    cl1 = -torch.mean(torch.log(pos_v / all_v))
    cl2 = -torch.mean(torch.sum(torch.log(1 - neg_v / all_v), dim=1))
    return cl1 + cl2

@torch.no_grad()
def sampling_positive_seqs(
    label_type: torch.Tensor,        # [B, L]
    label_time: torch.Tensor,        # [B, L]
    significance: torch.Tensor,      # [B, L]  越小越不重要
    pad_mask: torch.Tensor,
    pad_id,           # [B, L]  True=有效(非pad), False=pad
    ratio_remove: float = 0.2,       # 、
):
    assert label_type.shape == label_time.shape == significance.shape == pad_mask.shape
    B, L = label_type.shape
    device = label_type.device

    pos_type = torch.empty_like(label_type)
    pos_time = torch.empty_like(label_time)
    pos_mask = torch.zeros_like(pad_mask, dtype=torch.bool)

    # 预取一个 pad 填充值：优先用原序列里已有的 pad 位；否则用 0
    #（如果你的 pad id/time 有固定规范，也可以改成显式传入）
    def _pad_fill_vals(b):
        if (~pad_mask[b]).any():
            pad_type_val = label_type[b, ~pad_mask[b]][0]
            pad_time_val = label_time[b, ~pad_mask[b]][0]
        else:
            pad_type_val = torch.tensor(pad_id, dtype=label_type.dtype, device=device)
            pad_time_val = torch.tensor(float(pad_id), dtype=label_time.dtype, device=device)
        return pad_type_val, pad_time_val

    for b in range(B):
        valid_idx = torch.nonzero(pad_mask[b], as_tuple=False).squeeze(-1)   # 有效事件位置
        n_valid = valid_idx.numel()
        k_remove = int(math.floor(n_valid * ratio_remove))
        sig_valid = significance[b, valid_idx] 
        bottom_rel = torch.topk(sig_valid, k=k_remove, largest=False).indices
        remove_idx = valid_idx[bottom_rel]

        keep_mask_local = torch.ones(n_valid, dtype=torch.bool, device=device)
        if remove_idx.numel() > 0:
            rm_set = set(remove_idx.tolist())
            for i, vidx in enumerate(valid_idx.tolist()):
                if vidx in rm_set:
                    keep_mask_local[i] = False

        keep_idx = valid_idx[keep_mask_local]                                 # [k_keep]
        k_keep = keep_idx.numel()
        if k_keep > 0:
            keep_times = label_time[b, keep_idx]
            order = torch.argsort(keep_times, dim=0)                          # 升序
            keep_idx = keep_idx[order]

            pos_type[b, :k_keep] = label_type[b, keep_idx]
            pos_time[b, :k_keep] = label_time[b, keep_idx]
            pos_mask[b, :k_keep] = True
        pad_type_val, pad_time_val = _pad_fill_vals(b)
        if k_keep < L:
            pos_type[b, k_keep:] = pad_type_val
            pos_time[b, k_keep:] = pad_time_val
    return pos_type, pos_time, pos_mask
@torch.no_grad()
def sampling_negative_seqs_random(
    label_type: torch.Tensor,        # [B, L]
    label_time: torch.Tensor,        # [B, L]
    pad_mask: torch.Tensor,
    pad_id,          # [B, L]  True=有效(非pad), False=pad
    num_neg: int = 20,
    ratio_remove: float = 0.2,
):

    assert label_type.shape == label_time.shape == pad_mask.shape
    B, L = label_type.shape
    device = label_type.device

    # 结果容器（先按 [num_neg, B, L] 组织，最后 reshape 到 [B*num_neg, L]）
    neg_type = torch.empty((num_neg, B, L), dtype=label_type.dtype, device=device)
    neg_time = torch.empty((num_neg, B, L), dtype=label_time.dtype, device=device)
    neg_mask = torch.zeros((num_neg, B, L), dtype=torch.bool, device=device)

    # 取每条样本的 pad 填充值；若该样本没有 pad，则回退到 0
    def _pad_fill_vals(b):
        if (~pad_mask[b]).any():
            pad_type_val = label_type[b, ~pad_mask[b]][0]
            pad_time_val = label_time[b, ~pad_mask[b]][0]
        else:
            pad_type_val = torch.tensor(pad_id, dtype=label_type.dtype, device=device)
            pad_time_val = torch.tensor(float(pad_id), dtype=label_time.dtype, device=device)
        return pad_type_val, pad_time_val

    for b in range(B):
        valid_idx = torch.nonzero(pad_mask[b], as_tuple=False).squeeze(-1)   # 该样本有效位
        n_valid = valid_idx.numel()
        keep_ratio = max(0.0, min(1.0, 1.0 - ratio_remove))
        k_keep = int(math.floor(n_valid * keep_ratio))
        pad_type_val, pad_time_val = _pad_fill_vals(b)
        
        for n in range(num_neg):
            perm = torch.randperm(n_valid, device=device)
            keep_local = perm[:k_keep]
            keep_idx = valid_idx[keep_local]                                   # 全局下标 [k_keep]

            # 保留的事件按时间升序排列并前移
            if k_keep > 0:
                keep_times = label_time[b, keep_idx]
                order = torch.argsort(keep_times, dim=0)                       # 升序
                keep_idx = keep_idx[order]

                neg_type[n, b, :k_keep] = label_type[b, keep_idx]
                neg_time[n, b, :k_keep] = label_time[b, keep_idx]
                neg_mask[n, b, :k_keep] = True

            # 其余位置填 pad
            if k_keep < L:
                neg_type[n, b, k_keep:] = pad_type_val
                neg_time[n, b, k_keep:] = pad_time_val
                # neg_mask 剩余维持 False
    neg_type = neg_type.permute(1, 0, 2).reshape(B * num_neg, L)
    neg_time = neg_time.permute(1, 0, 2).reshape(B * num_neg, L)
    neg_mask = neg_mask.permute(1, 0, 2).reshape(B * num_neg, L)
    return neg_type, neg_time, neg_mask
def build_attn_mask_from_nonpad(non_pad_mask_bool: torch.Tensor) -> torch.Tensor:
    """
    non_pad_mask_bool: [B, L], True=非pad（有效），False=pad
    返回: attn_mask [B, L, L]，True 表示要屏蔽
    """
    B, L = non_pad_mask_bool.shape
    device = non_pad_mask_bool.device
    # 因果上三角
    subsequent = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).expand(B, -1, -1)
    # key padding：把 key 的 pad 列屏蔽
    key_pad = (~non_pad_mask_bool).unsqueeze(1).expand(-1, L, -1)
    return subsequent | key_pad

if __name__ == "__main__":
    event_time = torch.tensor([[ 2.6145,  3.3997,  4.1694,  4.8912,  5.6015,  6.3984,  7.1626,  7.9327,
          7.9443,  7.9585,  8.7026,  9.4657, 10.2362, 10.2504, 11.0384, 11.0492,
         11.7913, 12.5572, 13.3291, 13.3433, 14.0806, 14.8543, 15.6146, 15.6264,
         16.4235, 16.4378, 16.4479, 16.4597, 16.4713, 17.2141, 17.9598, 17.9711,
         17.9814, 17.9948, 18.7827, 18.7974, 19.5471, 20.2786, 20.2920, 21.0728,
         21.0832, 21.0974, 21.8809, 21.8954, 21.9103, 22.7035, 23.4497, 23.4601,
         23.4725, 24.2493, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000,
         16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000,
         16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000,
         16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000,
         16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000, 16.0000,
         16.0000, 16.0000]])
    event_type = torch.tensor([[ 1,  1,  9,  1,  2,  2,  7,  5,  2,  1,  1,  3,  1,  0,  0,  1,  4, 12,
          5,  1,  1,  9,  5,  1,  0,  1,  7,  3, 15,  0,  3, 13,  0,  8,  0, 12,
          0,  0,  1,  1,  7,  0,  4,  1,  0,  4,  0,  9,  3,  1, 16, 16, 16, 16,
         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
         16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]])
    significance = torch.tensor([[2.0082e+01, 1.5325e+01, 1.2756e+01, 1.1621e+01, 1.1757e+01, 1.3601e+01,
         1.6884e+01, 2.0638e+01, 1.9634e+01, 1.8836e+01, 2.0351e+01, 1.8145e+01,
         1.3582e+01, 1.3047e+01, 9.1205e+00, 8.7949e+00, 6.7431e+00, 5.9354e+00,
         5.9280e+00, 5.7589e+00, 6.0157e+00, 6.1091e+00, 5.7592e+00, 5.5775e+00,
         4.8611e+00, 4.7101e+00, 4.5717e+00, 4.4400e+00, 4.3152e+00, 3.6555e+00,
         3.1195e+00, 3.0399e+00, 2.9637e+00, 2.8884e+00, 2.5541e+00, 2.4938e+00,
         2.3084e+00, 2.1725e+00, 2.1241e+00, 1.9707e+00, 1.9259e+00, 1.8813e+00,
         1.7093e+00, 1.6683e+00, 1.6280e+00, 1.4816e+00, 1.4097e+00, 1.3765e+00,
         1.3439e+00, 1.3849e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
    non_mask = torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False]])
    sampling_negative_seqs_random(event_type,event_time, non_mask, 16)