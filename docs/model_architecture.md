# FastWAM 模型架构详细图

## 总览：完整模型结构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FastWAM 完整模型                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   VAE    │  │Proprio Encoder│  │ Video Expert │  │   Action Expert      │   │
│  │(Encoder) │  │  (Linear)    │  │ (WanVideoDiT)│  │   (ActionDiT)        │   │
│  │          │  │              │  │  30 layers   │  │    30 layers         │   │
│  │ frozen   │  │  trainable   │  │  trainable   │  │    trainable         │   │
│  └──────────┘  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                                           ↕ MoT 共享注意力 ↕                    │
│                                  ┌────────────────────────┐                    │
│                                  │  Mixture of Transformers │                    │
│                                  │   (混合注意力机制)       │                    │
│                                  └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. VAE 视频编码器 (Frozen, 不训练)

```
┌────────────────────────────────────────────────────────────────────┐
│                    WanVideoVAE (VideoVAE_)                          │
│                                                                    │
│  参数：z_dim=16, dim=96, dim_mult=[1, 2, 4, 4]                     │
│  空间下采样：8×  时间下采样：4×                                       │
│                                                                    │
│  输入: video [B, 3, 9, 224, 448]                                    │
│        (RGB, 9帧, 高224, 宽448)                                     │
│                                                                    │
│  ┌──────────────────────── Encoder3d ────────────────────────┐     │
│  │                                                           │     │
│  │  Conv3d(3→96)                                             │     │
│  │       ↓                                                   │     │
│  │  Stage 1: 2× ResBlock(96→96)     无时间下采样   无空间下采样│     │
│  │       ↓                                                   │     │
│  │  Stage 2: 2× ResBlock(96→192)    时间下采样×2   空间下采样×2│     │
│  │       ↓                                                   │     │
│  │  Stage 3: 2× ResBlock(192→384)   时间下采样×2   空间下采样×2│     │
│  │       ↓                                                   │     │
│  │  Stage 4: 2× ResBlock(384→384)   无时间下采样   空间下采样×2│     │
│  │       ↓                                                   │     │
│  │  Conv3d(384→32)                                           │     │
│  │                                                           │     │
│  └───────────────────────────────────────────────────────────┘     │
│       ↓                                                            │
│  Conv1x1(32→32) → split → μ [B,16,T_lat,28,56]                     │
│       ↓                                                            │
│  标准化: (μ - mean) / std                                           │
│                                                                    │
│  输出: latents [B, 16, 3, 28, 56]                                   │
│                                                                    │
│  ※ z_dim=16, 但 video_dit.in_dim=48                                │
│  ※ 实际 in_dim = z_dim * patch_size[0] * ... = 16*1*... 需要验证   │
│                                                                    │
│  注：由于 patch_size=[1,2,2], patchify 时会把                        │
│      [B,16,3,28,56] → Conv3d(16→3072, k=[1,2,2], s=[1,2,2])       │
│      → [B, 3072, 3, 14, 28] → reshape → [B, 3*14*28, 3072]       │
│      即 tokens = 1176 个 (每帧 14*28=392 个 token)                  │
│                                                                    │
│  实际 in_dim=48 意味着 VAE z_dim*3=48? 需要看具体加载               │
└────────────────────────────────────────────────────────────────────┘
```

**Shape 变化:**
```
video [B, 3, 9, 224, 448]
  → VAE Encoder
  → latents [B, 16, T_lat, H_lat, W_lat]
  
其中:
  T_lat = (9-1)/4 + 1 = 3  (时间下采样4×，首帧特殊处理)
  H_lat = 224/8 = 28
  W_lat = 448/8 = 56
  
→ latents [B, 16, 3, 28, 56]

注: config 中 in_dim=48, out_dim=48
实际模型可能使用 z_dim=16 但 scale 后 channel 数变为 48
```

---

## 2. Proprio Encoder (本体感知编码器)

```
┌────────────────────────────────────────────┐
│           Proprio Encoder                   │
│                                            │
│  结构: nn.Linear(8, 4096)                   │
│  就是一个线性层，没有其他                     │
│                                            │
│  输入: proprio [B, 8]                       │
│       (第0步的本体感知状态)                   │
│                                            │
│  处理:                                      │
│    proprio [B, 8]                           │
│         ↓ Linear(8 → 4096)                 │
│    proprio_token [B, 1, 4096]              │
│         ↓ cat到context末尾                  │
│    context [B, 128+1, 4096]                │
│    mask    [B, 128+1]                      │
│                                            │
│  作用: 把机器人当前状态编码为一个             │
│        "伪文本token"，拼入条件序列           │
└────────────────────────────────────────────┘
```

---

## 3. Video Expert (WanVideoDiT) - 30层

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Video Expert (WanVideoDiT)                            │
│                                                                          │
│  参数: hidden_dim=3072, num_heads=24, attn_head_dim=128                  │
│        ffn_dim=14336, num_layers=30, patch_size=[1,2,2]                  │
│        in_dim=48, out_dim=48, text_dim=4096, freq_dim=256                │
│                                                                          │
│ ┌──────────────────────── pre_dit() ────────────────────────────┐       │
│ │                                                                │       │
│ │  输入: x = latents [B, 48, 3, 28, 56]                         │       │
│ │         timestep [B] (扩散时间步)                              │       │
│ │         context [B, 129, 4096] (文本+proprio)                  │       │
│ │                                                                │       │
│ │  ① Patchify:                                                   │       │
│ │     Conv3d(48→3072, kernel=[1,2,2], stride=[1,2,2])           │       │
│ │     [B, 48, 3, 28, 56] → [B, 3072, 3, 14, 28]               │       │
│ │     → reshape → [B, 1176, 3072]                              │       │
│ │     (3帧 × 14×28=392 tokens/帧 = 1176 总token)               │       │
│ │                                                                │       │
│ │  ② Time Embedding (逐token独立timestep):                       │       │
│ │     首帧token的timestep=0 (干净), 其余帧=采样的t              │       │
│ │     sinusoidal(256) → Linear(256→3072) → SiLU                 │       │
│ │                      → Linear(3072→3072) = t_emb              │       │
│ │     → Linear(3072→3072×6) = t_mod [B, 1176, 6, 3072]         │       │
│ │       (shift_msa, scale_msa, gate_msa,                        │       │
│ │        shift_mlp, scale_mlp, gate_mlp)                        │       │
│ │                                                                │       │
│ │  ③ Text Embedding:                                             │       │
│ │     Linear(4096→3072) → GELU → Linear(3072→3072)             │       │
│ │     context [B, 129, 4096] → [B, 129, 3072]                  │       │
│ │                                                                │       │
│ │  ④ 3D RoPE:                                                    │       │
│ │     分别对 T, H, W 维度计算旋转位置编码                        │       │
│ │     freqs [1176, 1, 128] (head_dim=128)                       │       │
│ │                                                                │       │
│ │  输出: tokens [B, 1176, 3072]                                  │       │
│ │         + freqs, t_mod, context, context_mask                  │       │
│ └────────────────────────────────────────────────────────────────┘       │
│                              ↓                                           │
│                     进入 MoT 共享注意力 (见第5节)                          │
│                              ↓                                           │
│ ┌──────────────────────── post_dit() ───────────────────────────┐       │
│ │                                                                │       │
│ │  Head层:                                                       │       │
│ │    LayerNorm(3072) → AdaLN调制(shift,scale)                   │       │
│ │    → Linear(3072 → 48×1×2×2 = 192)                           │       │
│ │    tokens [B, 1176, 3072] → [B, 1176, 192]                   │       │
│ │                                                                │       │
│ │  Unpatchify:                                                   │       │
│ │    [B, 1176, 192] → reshape → [B, 48, 3, 28, 56]             │       │
│ │                                                                │       │
│ │  输出: pred_video [B, 48, 3, 28, 56]                           │       │
│ └────────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Action Expert (ActionDiT) - 30层

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Action Expert (ActionDiT)                             │
│                                                                          │
│  参数: hidden_dim=1024, num_heads=24, attn_head_dim=128                  │
│        ffn_dim=4096, num_layers=30                                       │
│        action_dim=7, text_dim=4096, freq_dim=256                         │
│                                                                          │
│ ┌──────────────────────── pre_dit() ────────────────────────────┐       │
│ │                                                                │       │
│ │  输入: action_tokens [B, 32, 7] (加噪后的动作)                 │       │
│ │         timestep [B]                                           │       │
│ │         context [B, 129, 4096]                                 │       │
│ │                                                                │       │
│ │  ① Action Embedding:                                           │       │
│ │     Linear(7 → 1024)                                          │       │
│ │     [B, 32, 7] → [B, 32, 1024]                               │       │
│ │                                                                │       │
│ │  ② Time Embedding:                                             │       │
│ │     sinusoidal(256) → Linear(256→1024) → SiLU                 │       │
│ │                      → Linear(1024→1024) = t_emb              │       │
│ │     → SiLU → Linear(1024→1024×6) = t_mod [B, 6, 1024]        │       │
│ │       (shift_msa, scale_msa, gate_msa,                        │       │
│ │        shift_mlp, scale_mlp, gate_mlp)                        │       │
│ │                                                                │       │
│ │  ③ Text Embedding:                                             │       │
│ │     Linear(4096→1024) → GELU → Linear(1024→1024)             │       │
│ │     context [B, 129, 4096] → [B, 129, 1024]                  │       │
│ │                                                                │       │
│ │  ④ 1D RoPE:                                                    │       │
│ │     对 32 步的时间位置编码                                     │       │
│ │     freqs [32, 1, 128]                                        │       │
│ │                                                                │       │
│ │  输出: tokens [B, 32, 1024]                                    │       │
│ │         + freqs, t_mod, context_emb, context_mask              │       │
│ └────────────────────────────────────────────────────────────────┘       │
│                              ↓                                           │
│                     进入 MoT 共享注意力 (见第5节)                          │
│                              ↓                                           │
│ ┌──────────────────────── post_dit() ───────────────────────────┐       │
│ │                                                                │       │
│ │  ActionHead:                                                   │       │
│ │    LayerNorm(1024) → AdaLN调制(shift,scale)                   │       │
│ │    → Linear(1024 → 7)                                         │       │
│ │    tokens [B, 32, 1024] → [B, 32, 7]                         │       │
│ │                                                                │       │
│ │  输出: pred_action [B, 32, 7]                                  │       │
│ └────────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. MoT 共享注意力（核心机制）- 30层

**两个 Expert 共用同一种 DiTBlock 结构，但各自有独立权重。**
**共享发生在 Self-Attention 的 Q/K/V 拼接阶段。**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     MoT Forward (每层的计算)                                   │
│                                                                              │
│  输入: video_tokens [B, 1176, 3072]                                          │
│        action_tokens [B, 32, 1024]                                           │
│                                                                              │
│  ╔══════════════════════════════════════════════════════════════════════════╗ │
│  ║                        第 i 层 (i = 0..29)                              ║ │
│  ╠══════════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                         ║ │
│  ║  ┌─── Video Expert Block[i] ───┐    ┌─── Action Expert Block[i] ───┐  ║ │
│  ║  │                             │    │                               │  ║ │
│  ║  │  AdaLN: norm + shift/scale  │    │  AdaLN: norm + shift/scale   │  ║ │
│  ║  │         ↓                   │    │         ↓                    │  ║ │
│  ║  │  Q = RoPE(RMSNorm(Wq·x))   │    │  Q = RoPE(RMSNorm(Wq·x))   │  ║ │
│  ║  │  K = RoPE(RMSNorm(Wk·x))   │    │  K = RoPE(RMSNorm(Wk·x))   │  ║ │
│  ║  │  V = Wv·x                   │    │  V = Wv·x                    │  ║ │
│  ║  │                             │    │                               │  ║ │
│  ║  │  Wq: 3072→3072 (24×128)    │    │  Wq: 1024→3072 (24×128)     │  ║ │
│  ║  │  Wk: 3072→3072             │    │  Wk: 1024→3072              │  ║ │
│  ║  │  Wv: 3072→3072             │    │  Wv: 1024→3072              │  ║ │
│  ║  │                             │    │                               │  ║ │
│  ║  │  Q_v [B,1176,3072]         │    │  Q_a [B,32,3072]             │  ║ │
│  ║  │  K_v [B,1176,3072]         │    │  K_a [B,32,3072]             │  ║ │
│  ║  │  V_v [B,1176,3072]         │    │  V_a [B,32,3072]             │  ║ │
│  ║  └─────────────┬───────────────┘    └──────────────┬────────────────┘  ║ │
│  ║                │                                    │                   ║ │
│  ║                ▼                                    ▼                   ║ │
│  ║  ┌──────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │         ★ 混合注意力 (Shared Attention Computation) ★            │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  Q_cat = [Q_v ; Q_a]  →  [B, 1208, 3072]                       │  ║ │
│  ║  │  K_cat = [K_v ; K_a]  →  [B, 1208, 3072]                       │  ║ │
│  ║  │  V_cat = [V_v ; V_a]  →  [B, 1208, 3072]                       │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  Attention = softmax(Q_cat · K_cat^T / √d + mask) · V_cat      │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  Attention Mask [1208 × 1208]:                                  │  ║ │
│  ║  │  ┌───────────────────────────┬──────────┐                       │  ║ │
│  ║  │  │  Video→Video              │Video→Act │                       │  ║ │
│  ║  │  │  (first_frame_causal)     │   ✗      │                       │  ║ │
│  ║  │  │  首帧看不到后续帧          │  blocked │                       │  ║ │
│  ║  │  │  后续帧能看所有帧          │          │                       │  ║ │
│  ║  │  ├───────────────────────────┼──────────┤                       │  ║ │
│  ║  │  │  Action→Video首帧         │ Act→Act  │                       │  ║ │
│  ║  │  │  只看前392个token(首帧)   │ 全连接   │                       │  ║ │
│  ║  │  │  ✓(首帧) ✗(后续帧)       │   ✓      │                       │  ║ │
│  ║  │  └───────────────────────────┴──────────┘                       │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  输出: mixed [B, 1208, 3072]                                    │  ║ │
│  ║  │        → split → video_mixed [B, 1176, 3072]                   │  ║ │
│  ║  │                   action_mixed [B, 32, 3072]                    │  ║ │
│  ║  └──────────────────────────────────────────────────────────────────┘  ║ │
│  ║                │                                    │                   ║ │
│  ║                ▼                                    ▼                   ║ │
│  ║  ┌─── Video Post-Attn ────────┐    ┌─── Action Post-Attn ──────────┐  ║ │
│  ║  │                             │    │                               │  ║ │
│  ║  │  Wo(mixed) [3072→3072]     │    │  Wo(mixed) [3072→1024]       │  ║ │
│  ║  │  Gate: x = x + gate * out  │    │  Gate: x = x + gate * out    │  ║ │
│  ║  │         ↓                   │    │         ↓                    │  ║ │
│  ║  │  Cross-Attention:           │    │  Cross-Attention:            │  ║ │
│  ║  │    Q: video_tokens          │    │    Q: action_tokens          │  ║ │
│  ║  │    K,V: text_context        │    │    K,V: text_context         │  ║ │
│  ║  │    [B,1176,3072]×[B,129,    │    │    [B,32,1024]×[B,129,      │  ║ │
│  ║  │     3072] → [B,1176,3072]  │    │     1024] → [B,32,1024]     │  ║ │
│  ║  │         ↓                   │    │         ↓                    │  ║ │
│  ║  │  FFN:                       │    │  FFN:                        │  ║ │
│  ║  │    AdaLN: norm+shift/scale  │    │    AdaLN: norm+shift/scale   │  ║ │
│  ║  │    Linear(3072→14336)       │    │    Linear(1024→4096)         │  ║ │
│  ║  │    GELU                     │    │    GELU                      │  ║ │
│  ║  │    Linear(14336→3072)       │    │    Linear(4096→1024)         │  ║ │
│  ║  │    Gate: x = x + gate*out   │    │    Gate: x = x + gate*out    │  ║ │
│  ║  │                             │    │                               │  ║ │
│  ║  └─────────────────────────────┘    └───────────────────────────────┘  ║ │
│  ║                │                                    │                   ║ │
│  ║                ▼                                    ▼                   ║ │
│  ║  video_tokens [B, 1176, 3072]      action_tokens [B, 32, 1024]         ║ │
│  ║                                                                         ║ │
│  ╚═══════════════════════════════════ 重复30层 ════════════════════════════╝ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. DiTBlock 内部结构（单层详细）

Video Expert 和 Action Expert 使用**同一种 DiTBlock 类**，但参数维度不同：

```
┌──────────────────────────────────────────────────────────────┐
│              DiTBlock (通用结构)                               │
│                                                              │
│  参数 (Video): hidden=3072, heads=24, head_dim=128, ffn=14336│
│  参数 (Action): hidden=1024, heads=24, head_dim=128, ffn=4096│
│                                                              │
│  modulation: Parameter [1, 6, hidden_dim]                    │
│                                                              │
│  输入: x [B, S, D], context, t_mod, freqs                    │
│                                                              │
│  ┌─────────── Self-Attention with AdaLN ──────────────────┐ │
│  │                                                         │ │
│  │  (modulation + t_mod) → chunk(6)                       │ │
│  │  → shift_msa, scale_msa, gate_msa,                    │ │
│  │    shift_mlp, scale_mlp, gate_mlp                      │ │
│  │                                                         │ │
│  │  x_norm = LayerNorm(x) * (1 + scale_msa) + shift_msa  │ │
│  │                                                         │ │
│  │  q = RoPE( RMSNorm( Linear(x_norm) ) )                │ │
│  │  k = RoPE( RMSNorm( Linear(x_norm) ) )                │ │
│  │  v = Linear(x_norm)                                    │ │
│  │                                                         │ │
│  │  attn_out = FlashAttention(q, k, v, mask)              │ │
│  │  attn_out = Linear(attn_out)  [Wo projection]          │ │
│  │                                                         │ │
│  │  x = x + gate_msa * attn_out   [残差 + 门控]           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────── Cross-Attention ────────────────────────────┐ │
│  │                                                         │ │
│  │  q = RMSNorm( Linear( LayerNorm(x) ) )                │ │
│  │  k = RMSNorm( Linear(context) )                        │ │
│  │  v = Linear(context)                                   │ │
│  │                                                         │ │
│  │  cross_out = FlashAttention(q, k, v, context_mask)     │ │
│  │  cross_out = Linear(cross_out)                         │ │
│  │                                                         │ │
│  │  x = x + cross_out   [残差，无门控]                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────── FFN with AdaLN ─────────────────────────────┐ │
│  │                                                         │ │
│  │  x_norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp  │ │
│  │                                                         │ │
│  │  ffn_out = Linear(x_norm, D→FFN_D)                    │ │
│  │          → GELU                                        │ │
│  │          → Linear(FFN_D→D)                             │ │
│  │                                                         │ │
│  │  x = x + gate_mlp * ffn_out   [残差 + 门控]            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  输出: x [B, S, D]                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 训练时完整数据流

```
═══════════════════════════════════════════════════════════════════════════════════
                         训练时完整数据流
═══════════════════════════════════════════════════════════════════════════════════

DataLoader 输出:
  video [B,3,9,224,448]  action [B,32,7]  proprio [B,32,8]  context [B,128,4096]

    │                         │                │                │
    ▼                         │                │                │
┌────────────┐                │                │                │
│ VAE Encode │                │                │                │
│ (frozen)   │                │                │                │
└─────┬──────┘                │                │                │
      │                       │                │                │
      ▼                       │                ▼                │
latents                       │          proprio[:,0,:]         │
[B,48,3,28,56]                │          = [B, 8]              │
      │                       │                │                │
      │                       │                ▼                │
      │                       │         ┌──────────────┐       │
      │                       │         │Proprio Encoder│       │
      │                       │         │ Linear(8→4096)│       │
      │                       │         └──────┬───────┘       │
      │                       │                │               │
      │                       │                ▼               ▼
      │                       │         cat → context [B, 129, 4096]
      │                       │                         mask [B, 129]
      │                       │                              │
      ▼                       ▼                              │
┌──────────┐           ┌──────────┐                          │
│ + noise  │           │ + noise  │                          │
│ t_video  │           │ t_action │                          │
│ (随机)    │           │ (随机)    │                          │
└────┬─────┘           └────┬─────┘                          │
     │                      │                                │
     ▼                      ▼                                ▼
noisy_latents          noisy_action                      context
[B,48,3,28,56]         [B,32,7]                      [B,129,4096]
     │                      │                                │
     ▼                      ▼                                │
┌────────────────┐   ┌────────────────┐                      │
│ Video Expert   │   │ Action Expert  │                      │
│   pre_dit()    │   │   pre_dit()    │                      │
│                │   │                │                      │
│ Patchify       │   │ Linear(7→1024) │                      │
│ +Time Embed    │   │ +Time Embed    │◄─────────────────────┘
│ +Text Embed    │◄──┤ +Text Embed    │
│ +3D RoPE       │   │ +1D RoPE       │
└───────┬────────┘   └───────┬────────┘
        │                    │
        ▼                    ▼
  video_tokens          action_tokens
  [B, 1176, 3072]      [B, 32, 1024]
        │                    │
        ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│              MoT: 30 层共享注意力                              │
│                                                              │
│  每层:                                                        │
│    1. 各Expert独立: Q/K/V投影 + RoPE + AdaLN调制             │
│    2. 拼接: [Q_v;Q_a] [K_v;K_a] [V_v;V_a]                   │
│    3. 联合注意力: FlashAttn(Q_cat, K_cat, V_cat, mask)       │
│    4. 拆分: video_out, action_out                            │
│    5. 各Expert独立: Cross-Attn(text) + FFN                   │
│                                                              │
│  注意力mask: Action只看Video首帧, Video看不到Action           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
  video_tokens          action_tokens
  [B, 1176, 3072]      [B, 32, 1024]
        │                    │
        ▼                    ▼
┌────────────────┐   ┌────────────────┐
│ Video Expert   │   │ Action Expert  │
│   post_dit()   │   │   post_dit()   │
│                │   │                │
│ Head + Unpatch │   │ Linear(1024→7) │
└───────┬────────┘   └───────┬────────┘
        │                    │
        ▼                    ▼
  pred_video            pred_action
  [B,48,3,28,56]        [B,32,7]
        │                    │
        ▼                    ▼
┌────────────────┐   ┌────────────────┐
│ MSE Loss       │   │ MSE Loss       │
│ (vs target)    │   │ (vs target)    │
│ × weight       │   │ × weight       │
└───────┬────────┘   └───────┬────────┘
        │                    │
        ▼                    ▼
   loss_video           loss_action
        │                    │
        └────────┬───────────┘
                 ▼
        loss = 1.0 × loss_video + 1.0 × loss_action


═══════════════════════════════════════════════════════════════════════════════════
```

---

## 8. 推理时数据流（infer_action，只预测动作）

```
═══════════════════════════════════════════════════════════════════════════════════
                         推理时数据流 (infer_action)
═══════════════════════════════════════════════════════════════════════════════════

输入: 1张当前图片 + 文本指令 + 当前proprio

input_image [1,3,H,W]    prompt "pick up..."    proprio [1,8]
       │                        │                     │
       ▼                        ▼                     ▼
┌────────────┐         ┌──────────────┐      ┌──────────────┐
│ VAE Encode │         │ T5 Encoder   │      │Proprio Encoder│
│ (1帧)      │         │ (预计算缓存)  │      │ Linear(8→4096)│
└─────┬──────┘         └──────┬───────┘      └──────┬───────┘
      │                       │                     │
      ▼                       ▼                     ▼
first_frame_latents      context              cat → context
[1, 48, 1, 28, 56]      [1,128,4096]          [1, 129, 4096]
      │                                            │
      ▼                                            │
┌──────────────────────────────────────────────────┤
│ Video Expert pre_dit (首帧, timestep=0)          │
│ → video_tokens [1, 392, 3072]                    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│ prefill_video_cache()                                 │
│ 只跑 Video Expert 部分，缓存每层的 K_v, V_v          │
│ → kv_cache: 30层 × {K: [1,392,3072], V: [1,392,3072]}│
└──────────────────────┬───────────────────────────────┘
                       │
                       │  (缓存不变，重复使用)
                       │
   ┌───────────────────┼───────────────────────────────┐
   │                   │        去噪循环 (20步)         │
   │                   ▼                                │
   │  noise [1,32,7] (初始纯噪声)                      │
   │         │                                          │
   │         ▼ ──────── 每步 ──────────                │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ Action Expert pre_dit:                       │  │
   │  │   noisy_action → tokens [1, 32, 1024]       │  │
   │  │                                              │  │
   │  │ Mixed Attention (with cache):                │  │
   │  │   Q = Q_action [1, 32, 3072]                 │  │
   │  │   K = [K_video_cached ; K_action]            │  │
   │  │       [1, 392+32, 3072]                      │  │
   │  │   V = [V_video_cached ; V_action]            │  │
   │  │       [1, 392+32, 3072]                      │  │
   │  │   → action只看video首帧(前392) + 自身         │  │
   │  │                                              │  │
   │  │ Action Expert post_dit:                      │  │
   │  │   → pred_noise [1, 32, 7]                    │  │
   │  │                                              │  │
   │  │ Scheduler step:                              │  │
   │  │   latents = latents - delta * pred_noise     │  │
   │  └─────────────────────────────────────────────┘  │
   │         │                                          │
   │         ▼ (重复20步)                               │
   │                                                    │
   └────────────────────────────────────────────────────┘
                       │
                       ▼
              clean_action [1, 32, 7]
              (去噪后的干净动作序列)
              
═══════════════════════════════════════════════════════════════════════════════════
```

---

## 9. 参数规模汇总

| 模块 | 关键参数 | 估算参数量 |
|------|---------|-----------|
| **VAE** | z_dim=16, 4 stages | ~100M (frozen) |
| **T5 Text Encoder** | dim=4096, 24层, 64heads | ~4.7B (frozen, 预计算) |
| **Proprio Encoder** | Linear(8→4096) | 32K |
| **Video Expert** | dim=3072, 30层, ffn=14336, 24heads | ~4.5B |
| **Action Expert** | dim=1024, 30层, ffn=4096, 24heads | ~1.2B |
| **总训练参数** | Video Expert + Action Expert + Proprio | ~5.7B |

---

## 10. 两个 Expert 的对比

| 属性 | Video Expert | Action Expert |
|------|-------------|---------------|
| 类名 | WanVideoDiT | ActionDiT |
| 层数 | 30 | 30 |
| Hidden dim | 3072 | 1024 |
| Attention heads | 24 | 24 |
| Head dim | 128 | 128 |
| Attn hidden (heads×head_dim) | 3072 | 3072 |
| FFN dim | 14336 | 4096 |
| 输入 | latents [B,48,T,H,W] | action [B,32,7] |
| Token化 | Conv3d patch [1,2,2] | Linear(7→1024) |
| Token 数 | 1176 (3帧×14×28) | 32 |
| 位置编码 | 3D RoPE (T,H,W) | 1D RoPE |
| 输出投影 | Linear(3072→192) + unpatch | Linear(1024→7) |
| Text embed | Linear(4096→3072)→GELU→Linear | Linear(4096→1024)→GELU→Linear |
| Time embed | sin(256)→Linear→SiLU→Linear→SiLU→Linear(×6) | 同左 |
| 来源 | Wan2.2-TI2V-5B 预训练 | 独立预训练 |
