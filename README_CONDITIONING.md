# FastWAM Model Conditioning Documentation

Welcome! This directory contains comprehensive documentation about how the Wan2.2 FastWAM model handles image/video conditioning for diffusion-based video and action generation.

## 📚 Documentation Files

### 1. **QUICK_REFERENCE.md** ⭐ START HERE
- **Best for**: Quick understanding in 5-10 minutes
- **Contains**: 
  - TL;DR overview of conditioning mechanism
  - Complete data flow from dataset to loss
  - Key code locations with line numbers
  - Common misconceptions addressed
  - Performance optimization tips
  - Debugging checklist
- **Read this if**: You want a fast overview or need to quickly find a specific code location

### 2. **CONDITIONING_ANALYSIS.md** 📖 DETAILED REFERENCE
- **Best for**: Deep understanding of architecture and design decisions
- **Contains**:
  - Section 1: First frame encoding and injection mechanism (with VAE flow)
  - Section 2: MoT (Mixture of Transformers) design and attention
  - Section 3: Complete training data flow from dataset to loss
  - Section 4: Conditioning mechanisms analysis (what exists, what doesn't)
  - Section 5: Key design insights and rationale
  - Section 6: Inference modes (joint vs action-only)
  - Section 7: Complete condition flow diagram
  - Section 8: File references
- **Read this if**: You need to understand WHY the system is designed this way and need comprehensive explanations

### 3. **ARCHITECTURE_DIAGRAMS.md** 🎨 VISUAL GUIDE
- **Best for**: Visual learners who want to see the flow and connectivity
- **Contains**:
  - End-to-end training pipeline diagram
  - Frozen first frame conditioning mechanism (step-by-step)
  - MoT mixed attention mechanism visualization
  - Attention pattern connectivity diagram
  - Per-token timestep assignment explanation
  - Knowledge insulation in action loss (with gradient flow)
  - Inference modes comparison (joint vs action-only)
  - Loss computation flow with example tensors
- **Read this if**: You prefer visual explanations or need to present the architecture to others

### 4. **model_architecture.md** 🏗️ IMPLEMENTATION DETAILS
- **Best for**: Implementation-level understanding
- **Contains**:
  - VAE encoder specifications and shape transformations
  - Proprio encoder implementation
  - Video Expert (WanVideoDiT) architecture with parameter counts
  - Action Expert (ActionDiT) architecture with parameter counts
  - DiTBlock internal structure details
  - Complete training step data flow with exact tensor shapes
  - Inference time data flow (action-only)
  - Parameter scale summary
  - Expert comparison table
- **Read this if**: You need to implement changes, optimize memory, or debug tensor shape issues

---

## 🎯 Quick Navigation

### I want to understand...

#### The First Frame Conditioning Mechanism
- Quick overview: **QUICK_REFERENCE.md** → "How First Frame Conditioning Works"
- Detailed flow: **CONDITIONING_ANALYSIS.md** → "Section 1"
- Visual flow: **ARCHITECTURE_DIAGRAMS.md** → "Section 2"

#### The MoT Architecture
- Quick overview: **QUICK_REFERENCE.md** → "MoT Architecture"
- Detailed design: **CONDITIONING_ANALYSIS.md** → "Section 2"
- Visual mechanism: **ARCHITECTURE_DIAGRAMS.md** → "Section 3 & 4"

#### How Training Works
- Quick flow: **QUICK_REFERENCE.md** → "Complete Data Flow"
- Detailed breakdown: **CONDITIONING_ANALYSIS.md** → "Section 3"
- Visual flow: **ARCHITECTURE_DIAGRAMS.md** → "Section 1 & 8"

#### A Specific Code Location
- Use: **QUICK_REFERENCE.md** → "Key Code Locations"
- Then refer to original file for context

#### How Inference Works
- Quick guide: **QUICK_REFERENCE.md** → "Performance Tips"
- Detailed modes: **CONDITIONING_ANALYSIS.md** → "Section 6"
- Visual comparison: **ARCHITECTURE_DIAGRAMS.md** → "Section 7"

#### Why the Design is This Way
- Read: **CONDITIONING_ANALYSIS.md** → "Section 5: Key Design Insights"

---

## 🔑 Key Concepts at a Glance

### The Three "Anchors" of FastWAM Conditioning

1. **Frozen First Frame** (Hard Constraint)
   - First frame is encoded to latent space and frozen throughout training
   - Never included in the denoising loss
   - Prevents model from forgetting initial condition
   
2. **Per-Token Timestep** (Semantic Signal)
   - First frame tokens get t=0 (clean reference)
   - Future frames get t=t_sample (noisy targets)
   - Teaches model the distinction between condition and target
   
3. **Attention Mask** (Information Control)
   - Action tokens can only see first frame video tokens
   - Video tokens never see action tokens
   - Prevents information leakage between tasks

---

## 🏗️ File Structure

```
FastWAM/
├── src/fastwam/
│   ├── models/wan22/
│   │   ├── fastwam.py          # Main model (build_inputs, training_loss, etc.)
│   │   ├── mot.py              # Mixture of Transformers
│   │   ├── wan_video_dit.py    # Video expert
│   │   ├── action_dit.py       # Action expert
│   │   └── visual_encoder.py   # VAE/visual encoder wrapper
│   ├── trainer.py              # Training loop orchestration
│   └── ...
├── QUICK_REFERENCE.md          # 5-minute overview (you are here)
├── CONDITIONING_ANALYSIS.md    # Detailed 8-section analysis
├── ARCHITECTURE_DIAGRAMS.md    # Visual diagrams (8 sections)
├── model_architecture.md       # Implementation details (Chinese)
└── docs/
    ├── model_architecture.md
    ├── training_pipeline.md
    └── ...
```

---

## ⚡ Common Tasks

### Debugging "First frame not conditioned"
1. Check: `latents[:, :, 0:1] = inputs["first_frame_latents"]` in fastwam.py line 541
2. Verify: `token_timesteps[:, 0, :] = 0` in wan_video_dit.py line 546
3. Confirm: `pred_video = pred_video[:, :, 1:]` in loss computation (fastwam.py line 421)

### Understanding attention flow
1. Review: ARCHITECTURE_DIAGRAMS.md → "Section 4: Attention Pattern Visualization"
2. Check: Attention mask construction in fastwam.py lines 460-481
3. Trace: Mixed attention in mot.py lines 544-612

### Optimizing inference
1. Read: QUICK_REFERENCE.md → "Performance Tips"
2. Use: `infer_action()` instead of `infer_joint()` for action-only (10-20x faster)
3. Enable: Gradient checkpointing for 40% memory saving (10% speed cost)

### Adding extra conditioning images
1. Problem: Currently not supported (model designed for single-frame conditioning)
2. Required changes:
   - Encode additional images to latent space
   - Create separate expert or cross-attention pathway
   - Extend attention mask to include conditioning images
   - Modify context building to embed conditioning images
   - Adjust loss to only predict non-conditioned frames
3. Reference: CONDITIONING_ANALYSIS.md → "Section 4.3"

---

## 📊 Model Scale

| Component | Layers | Hidden Dim | Params | Status |
|-----------|--------|-----------|--------|--------|
| VAE Encoder | - | 16 channels | ~100M | Frozen |
| Text Encoder (T5) | 24 | 4096 | ~4.7B | Frozen (cached) |
| Video Expert | 30 | 3072 | ~4.5B | **Trainable** |
| Action Expert | 30 | 1024 | ~1.2B | **Trainable** |
| MoT Shared | 30 shared | 3072 | Counted above | **Trainable** |
| Total Trainable | - | - | ~5.7B | - |

---

## 🎓 Learning Path

### Level 1: High-Level Understanding
1. Read: QUICK_REFERENCE.md (20 minutes)
2. Skim: ARCHITECTURE_DIAGRAMS.md sections 1-4 (10 minutes)
3. **Total: 30 minutes** → You understand the basic mechanism

### Level 2: Implementation Understanding
1. Read: CONDITIONING_ANALYSIS.md sections 1-3 (30 minutes)
2. Read: model_architecture.md (30 minutes)
3. Browse original code with section references (20 minutes)
4. **Total: 80 minutes** → You can implement modifications

### Level 3: Expert Level
1. Deep read: All sections of CONDITIONING_ANALYSIS.md (60 minutes)
2. Study: ARCHITECTURE_DIAGRAMS.md with detailed notes (40 minutes)
3. Code-reading session: Trace through all 8 files systematically (120 minutes)
4. Experiment: Small modifications to verify understanding (60 minutes)
5. **Total: 280 minutes (4.5 hours)** → You can architect new features

---

## ❓ FAQ

**Q: Is there a way to condition on multiple images?**
A: No, the current architecture only supports single first-frame conditioning. See CONDITIONING_ANALYSIS.md Section 4.3 for how to extend it.

**Q: Why can't actions see future video frames?**
A: Because it would create an information leak – actions would know about future video states that they're supposed to help generate. This makes the problem ill-posed.

**Q: What does "frozen first frame" mean exactly?**
A: It means the first frame latents are never modified by the video denoising loss. They're computed once and used as a hard constraint. See ARCHITECTURE_DIAGRAMS.md Section 2 for the step-by-step process.

**Q: How fast is action-only inference?**
A: About 10-20x faster than joint inference because video K/V are precomputed and reused. See ARCHITECTURE_DIAGRAMS.md Section 7 for details.

**Q: Can I train without per-token timestep?**
A: Technically yes, but it would hurt performance. The per-token timestep is a key signal that tells the model "frame 0 is clean, frames 1+ are noisy". See QUICK_REFERENCE.md "Design Constraints" for why this matters.

---

## 🔗 Related Documentation

- Original paper: [Wan2.2 Research Paper]
- Training pipeline details: `docs/training_pipeline.md`
- Model configuration: `config/wan22_config.yaml`
- Inference examples: `examples/inference.py`

---

## 📝 Document Versions

| Document | Last Updated | Version |
|----------|-------------|---------|
| QUICK_REFERENCE.md | 2026-05-10 | 1.0 |
| CONDITIONING_ANALYSIS.md | 2026-05-10 | 1.0 |
| ARCHITECTURE_DIAGRAMS.md | 2026-05-10 | 1.0 |
| model_architecture.md | Earlier | 1.0 |

---

## 💡 Tips for Best Learning

1. **Start with QUICK_REFERENCE.md** - Get the big picture first
2. **Refer to ARCHITECTURE_DIAGRAMS.md** - Visualize as you read
3. **Deep dive into CONDITIONING_ANALYSIS.md** - Understand the why
4. **Check model_architecture.md** - Get implementation details
5. **Read the code** - See the real thing with line references
6. **Debug with checklists** - Use QUICK_REFERENCE.md debugging section

---

**Questions or corrections?** Please update these docs to help the next person!
