import torch
import math
from itertools import combinations
import random

def _top_indices_by_significance(event_type: torch.Tensor, significance: torch.Tensor, mask:torch.Tensor, frac: float = 0.4):
    B, L = event_type.shape
    top_idx_list = []
    for b in range(B):
        mask_ = mask[b]
        valid_idx = torch.nonzero(mask_, as_tuple=False).squeeze(1)  # 这些是全局位置索引 [l_b]
        if valid_idx.numel() == 0:
            top_idx_list.append(torch.empty(0, dtype=torch.long, device=event_type.device))
            continue

        sig_valid = significance[b, valid_idx]
        k = max(1, math.ceil(sig_valid.numel() * frac))
        # 取 top-k（降序）
        top_local = torch.topk(sig_valid, k=k, largest=True).indices
        top_idx = valid_idx[top_local]
        top_idx_list.append(top_idx)
    return top_idx_list

def _bottom_indices_by_significance(
    event_type: torch.Tensor,
    significance: torch.Tensor,
    mask: torch.Tensor,
    frac: float = 0.4
):
    B, L = event_type.shape
    bottom_idx_list = []

    for b in range(B):
        valid_mask = mask[b]             # True 表示有效事件
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            bottom_idx_list.append(torch.empty(0, dtype=torch.long, device=event_type.device))
            continue

        sig_valid = significance[b, valid_idx]
        k = max(1, math.ceil(sig_valid.numel() * frac))
        # 取 bottom-k（升序）
        bottom_local = torch.topk(sig_valid, k=k, largest=False).indices
        bottom_idx = valid_idx[bottom_local]
        bottom_idx_list.append(bottom_idx)

    return bottom_idx_list

def _all_pairs_from_indices(idx_tensor: torch.Tensor, max_pairs: int = None):
    idx_list = idx_tensor.tolist()
    pairs = list(combinations(idx_list, 2))  # 所有无序对
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    #这里已经shuffle过了
    random.shuffle(pairs)
    return pairs


def _apply_swaps_on_types(event_type_row: torch.Tensor, pairs_to_swap):
    """
    就地在一行 event_type 上执行交换（只交换 type，不动 time）
    """
    for (i, j) in pairs_to_swap:
        tmp = event_type_row[i].item()
        event_type_row[i] = event_type_row[j]
        event_type_row[j] = tmp
    return event_type_row


@torch.no_grad()
def make_positive_by_swaps(event_type: torch.Tensor,
                           event_time: torch.Tensor,
                           significance: torch.Tensor,
                           mask: torch.Tensor,
                           frac_low: float = 0.4,
                           swap_factor: float = 0.5,
                           max_pairs: int = 20000):
    B, L = event_type.shape
    pos_type = event_type.clone()
    pos_time = event_time.clone()  # 时间不动
    bottom_idx_list = _bottom_indices_by_significance(event_type, significance, mask, frac=frac_low)
    for b in range(B):
        k = bottom_idx_list[b].numel()
        total_pairs = k * (k - 1) // 2
        pairs = _all_pairs_from_indices(bottom_idx_list[b], max_pairs=max_pairs)
        swaps_pos = max(1, int(total_pairs * swap_factor))
        swaps_pos = min(swaps_pos, len(pairs))  # 受限于可用对数，主要是防止 max_pairs 的影响。total_pairs*swap_factor 如果>20000，虽然极不可能， 但是还是会起到作用。

        choose_pairs = pairs[:swaps_pos]
        _apply_swaps_on_types(pos_type[b], choose_pairs)
    return pos_type, pos_time


@torch.no_grad()
def make_negatives_by_swaps(event_type: torch.Tensor,
                            event_time: torch.Tensor,
                            significance: torch.Tensor,
                            mask: torch.Tensor,
                            num_neg: int = 20,
                            frac_top: float = 0.4,
                            max_pairs: int = 20000):
    assert num_neg >= 1
    B, L = event_type.shape
    neg_types = []
    neg_times = []
    masks = []
    top_idx_list = _top_indices_by_significance(event_type, significance,mask, frac=frac_top)

    for b in range(B):
        base_type = event_type[b].clone()
        base_time = event_time[b].clone()
        base_mask = mask[b].clone()
        k = top_idx_list[b].numel()
        total_pairs = k * (k - 1) // 2
        #significance 最大的对应的 event_type 的 id 的 pairs
        all_pairs = _all_pairs_from_indices(top_idx_list[b], max_pairs=max_pairs)
        swaps_per_neg = max(1, math.floor(total_pairs / num_neg))
        ptr = 0
        for _ in range(num_neg):
            et = base_type.clone()
            # 选本轮的 pairs
            end = ptr + swaps_per_neg
            pairs_this = all_pairs[ptr:end]
            ptr = end
            _apply_swaps_on_types(et, pairs_this)
            neg_types.append(et)
            neg_times.append(base_time)
            masks.append(base_mask)
    neg_type = torch.stack(neg_types, dim=0)
    neg_time = torch.stack(neg_times, dim=0)
    neg_mask = torch.stack(masks, dim=0)
    return neg_type, neg_time, neg_mask

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
    make_negatives_by_swaps(event_type,event_time, significance, non_mask)
    
