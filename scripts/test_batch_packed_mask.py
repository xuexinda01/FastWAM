"""Standalone unit test for batched-packed attention mask.

Reimplements the function logic directly so we don't pay the cost of
importing the full Wan2.2 / HFastWAM module chain (which loads ~5 GB
of model code at import time).
"""
from __future__ import annotations

import torch


def build_language_rows(task_len: int, subtask_len: int, device) -> torch.Tensor:
    """Mirror of LanguageExpert.build_language_rows — causal task block."""
    S = task_len + subtask_len
    rows = torch.zeros(S, S, dtype=torch.bool, device=device)
    if task_len > 0:
        rows[0:task_len, 0:task_len] = torch.tril(
            torch.ones((task_len, task_len), dtype=torch.bool, device=device)
        )
    rows[task_len:S, 0:task_len] = True
    for i in range(subtask_len):
        rows[task_len + i, task_len:task_len + i + 1] = True
    return rows


def build_video_to_video_mask_first_frame(
    video_seq_len: int, video_tokens_per_frame: int, device
) -> torch.Tensor:
    """Mirror of WanVideoDiT first_frame_causal mode (the production setting)."""
    m = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
    first = min(video_tokens_per_frame, video_seq_len)
    m[:first, first:] = False
    return m


def build_batched_packed_mask(
    lang_lens, video_seq_len, action_seq_len, video_tokens_per_frame, device,
):
    """**Direct copy** of HFastWAM._build_batched_packed_attention_mask logic."""
    N = len(lang_lens)
    if N == 0:
        raise ValueError("lang_lens must be non-empty")
    S_vid = int(video_seq_len)
    S_act = int(action_seq_len)
    lang_total = int(sum(lang_lens))
    video_total = N * S_vid
    action_total = N * S_act
    total = lang_total + video_total + action_total
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)

    lang_ranges = []
    offset = 0
    for L in lang_lens:
        lang_ranges.append((offset, offset + int(L)))
        offset += int(L)

    def vrange(i):
        s = lang_total + i * S_vid
        return s, s + S_vid

    def arange_(i):
        s = lang_total + video_total + i * S_act
        return s, s + S_act

    first_frame_tokens = min(int(video_tokens_per_frame), S_vid)

    for i in range(N):
        l_s, l_e = lang_ranges[i]
        L_i = l_e - l_s
        if L_i > 0:
            mask[l_s:l_e, l_s:l_e] = build_language_rows(L_i, 0, device)
            if S_vid > 0:
                v_s, _ = vrange(i)
                mask[l_s:l_e, v_s:v_s + first_frame_tokens] = True
        if S_vid > 0:
            v_s, v_e = vrange(i)
            if L_i > 0:
                mask[v_s:v_e, l_s:l_e] = True
            mask[v_s:v_e, v_s:v_e] = build_video_to_video_mask_first_frame(
                S_vid, video_tokens_per_frame, device
            )
        if S_act > 0:
            a_s, a_e = arange_(i)
            if L_i > 0:
                mask[a_s:a_e, l_s:l_e] = True
            if S_vid > 0:
                v_s, _ = vrange(i)
                mask[a_s:a_e, v_s:v_s + first_frame_tokens] = True
            mask[a_s:a_e, a_s:a_e] = True
    return mask


def test_shape():
    lang_lens = [5, 7]
    S_v, S_a, V_TPF = 9, 4, 3
    N = 2
    total = sum(lang_lens) + N * S_v + N * S_a  # 12 + 18 + 8 = 38
    mask = build_batched_packed_mask(lang_lens, S_v, S_a, V_TPF, torch.device("cpu"))
    assert mask.shape == (total, total), f"shape={mask.shape} != ({total},{total})"
    print(f"[OK] test_shape: {total}x{total}")


def test_block_diagonal_no_cross_sample():
    lang_lens = [5, 7]
    S_v, S_a, V_TPF = 9, 4, 3
    N = 2
    lang_total = sum(lang_lens)
    video_total = N * S_v
    mask = build_batched_packed_mask(lang_lens, S_v, S_a, V_TPF, torch.device("cpu"))

    # Per-sample full token sets.
    sample_tokens = []
    offset = 0
    for i, L in enumerate(lang_lens):
        toks = list(range(offset, offset + L))
        offset += L
        toks += list(range(lang_total + i * S_v, lang_total + (i + 1) * S_v))
        toks += list(range(lang_total + video_total + i * S_a, lang_total + video_total + (i + 1) * S_a))
        sample_tokens.append(set(toks))

    failures = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c].item():
                continue
            r_sample = next(i for i, s in enumerate(sample_tokens) if r in s)
            c_sample = next(i for i, s in enumerate(sample_tokens) if c in s)
            if r_sample != c_sample:
                failures += 1
                if failures <= 3:
                    print(f"  CROSS-SAMPLE LEAK: mask[{r},{c}]=True (sample {r_sample}->sample {c_sample})")
    assert failures == 0, f"{failures} cross-sample attention edges found"
    print(f"[OK] test_block_diagonal_no_cross_sample (0 leaks)")


def test_lang_causal_within_sample():
    lang_lens = [4, 6]
    mask = build_batched_packed_mask(lang_lens, 6, 0, 2, torch.device("cpu"))
    # Sample 0 lang: [0:4] x [0:4] — lower triangular.
    s0 = mask[0:4, 0:4]
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(s0, expected), f"sample0 lang not causal:\n{s0.int()}"
    s1 = mask[4:10, 4:10]
    expected = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    assert torch.equal(s1, expected), "sample1 lang not causal"
    print("[OK] test_lang_causal_within_sample")


def test_video_first_frame_attends_back():
    lang_lens = [4, 6]
    S_v, V_TPF = 6, 2
    lang_total = sum(lang_lens)  # 10
    mask = build_batched_packed_mask(lang_lens, S_v, 0, V_TPF, torch.device("cpu"))
    # Sample 0 video [10:16] -> sample 0 lang [0:4]: all True.
    block = mask[10:16, 0:4]
    assert torch.all(block).item(), "video0 should attend to lang0"
    # Sample 0 video -> sample 1 lang [4:10]: all False.
    block = mask[10:16, 4:10]
    assert not torch.any(block).item(), "video0 should NOT attend to lang1"
    # Sample 1 video [16:22] -> sample 1 lang [4:10]: all True.
    block = mask[16:22, 4:10]
    assert torch.all(block).item(), "video1 should attend to lang1"
    print("[OK] test_video_first_frame_attends_back")


def test_b1_special_case():
    """B=1 should produce a sensible mask (no degenerate cases)."""
    lang_lens = [5]
    mask = build_batched_packed_mask(lang_lens, 6, 0, 2, torch.device("cpu"))
    total = 5 + 6
    assert mask.shape == (total, total)
    # Lang causal
    assert torch.equal(mask[0:5, 0:5], torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    # Lang->first_frame True
    assert torch.all(mask[0:5, 5:7]).item()
    # Lang->later_frame False
    assert not torch.any(mask[0:5, 7:11]).item()
    print("[OK] test_b1_special_case")


def test_b3_three_samples():
    lang_lens = [3, 5, 4]
    S_v, S_a, V_TPF = 6, 2, 3
    mask = build_batched_packed_mask(lang_lens, S_v, S_a, V_TPF, torch.device("cpu"))
    N = 3
    lang_total = sum(lang_lens)  # 12
    video_total = N * S_v  # 18
    total = lang_total + video_total + N * S_a  # 12+18+6=36
    assert mask.shape == (total, total)
    # Sanity: action of sample 1 [12+18+2:12+18+4] = [32:34] should attend to
    # sample 1 lang [3:8] but NOT sample 0 lang [0:3] or sample 2 lang [8:12].
    assert torch.all(mask[32:34, 3:8]).item()
    assert not torch.any(mask[32:34, 0:3]).item()
    assert not torch.any(mask[32:34, 8:12]).item()
    print("[OK] test_b3_three_samples")


if __name__ == "__main__":
    test_shape()
    test_block_diagonal_no_cross_sample()
    test_lang_causal_within_sample()
    test_video_first_frame_attends_back()
    test_b1_special_case()
    test_b3_three_samples()
    print("\nAll mask tests passed ✓")
