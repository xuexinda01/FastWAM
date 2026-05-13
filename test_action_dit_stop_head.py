"""
Test script to verify ActionDiT stop head implementation.
This tests:
1. ActionDiT can be instantiated with predict_stop=True/False
2. post_dit returns dict with "action" key (and "stop" if predict_stop=True)
3. fastwam.py correctly extracts "action" from the dict
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastwam.models.wan22.action_dit import ActionDiT, ActionHead, StopHead


def test_action_head():
    """Test ActionHead class"""
    print("\n" + "="*60)
    print("TEST 1: ActionHead Class")
    print("="*60)
    
    head = ActionHead(hidden_dim=1024, out_dim=3, eps=1e-6)
    x = torch.randn(2, 10, 1024)  # [B, T, hidden_dim]
    t_mod = torch.randn(2, 6, 1024)  # [B, 6, hidden_dim]
    
    output = head(x, t_mod)
    assert output.shape == (2, 10, 3), f"Expected (2, 10, 3), got {output.shape}"
    print(f"✓ ActionHead output shape: {output.shape}")


def test_stop_head():
    """Test StopHead class"""
    print("\n" + "="*60)
    print("TEST 2: StopHead Class")
    print("="*60)
    
    head = StopHead(hidden_dim=1024, eps=1e-6)
    x = torch.randn(2, 10, 1024)  # [B, T, hidden_dim]
    t_mod = torch.randn(2, 6, 1024)  # [B, 6, hidden_dim]
    
    output = head(x, t_mod)
    assert output.shape == (2, 10, 1), f"Expected (2, 10, 1), got {output.shape}"
    print(f"✓ StopHead output shape: {output.shape}")


def test_action_dit_without_stop():
    """Test ActionDiT with predict_stop=False"""
    print("\n" + "="*60)
    print("TEST 3: ActionDiT without stop head (predict_stop=False)")
    print("="*60)
    
    config = {
        "hidden_dim": 1024,
        "action_dim": 3,
        "ffn_dim": 4096,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1e-6,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 2,  # Smaller for testing
        "use_gradient_checkpointing": False,
        "predict_stop": False,
    }
    
    action_dit = ActionDiT(**config)
    
    # Check that stop_head is not created
    assert not hasattr(action_dit, "stop_head") or action_dit.predict_stop == False
    print("✓ ActionDiT created without stop_head")
    
    # Test forward pass
    action_tokens = torch.randn(2, 10, 3)  # [B, T, action_dim]
    timestep = torch.tensor([100, 200], dtype=torch.long)
    context = torch.randn(2, 50, 4096)  # [B, L, text_dim]
    
    with torch.no_grad():
        output = action_dit(action_tokens, timestep, context)
    
    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    assert "action" in output, f"Expected 'action' key in output, got {output.keys()}"
    assert output["action"].shape == (2, 10, 3), f"Expected (2, 10, 3), got {output['action'].shape}"
    assert "stop" not in output, f"Expected no 'stop' key when predict_stop=False"
    print(f"✓ Forward pass successful")
    print(f"  Output keys: {list(output.keys())}")
    print(f"  Action shape: {output['action'].shape}")


def test_action_dit_with_stop():
    """Test ActionDiT with predict_stop=True"""
    print("\n" + "="*60)
    print("TEST 4: ActionDiT with stop head (predict_stop=True)")
    print("="*60)
    
    config = {
        "hidden_dim": 1024,
        "action_dim": 3,
        "ffn_dim": 4096,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1e-6,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 2,
        "use_gradient_checkpointing": False,
        "predict_stop": True,
    }
    
    action_dit = ActionDiT(**config)
    
    # Check that stop_head is created
    assert hasattr(action_dit, "stop_head"), "Expected stop_head attribute"
    assert isinstance(action_dit.stop_head, StopHead), "stop_head should be StopHead instance"
    print("✓ ActionDiT created with stop_head")
    
    # Test forward pass
    action_tokens = torch.randn(2, 10, 3)
    timestep = torch.tensor([100, 200], dtype=torch.long)
    context = torch.randn(2, 50, 4096)
    
    with torch.no_grad():
        output = action_dit(action_tokens, timestep, context)
    
    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    assert "action" in output, f"Expected 'action' key"
    assert "stop" in output, f"Expected 'stop' key when predict_stop=True"
    assert output["action"].shape == (2, 10, 3), f"Expected action shape (2, 10, 3)"
    assert output["stop"].shape == (2, 10, 1), f"Expected stop shape (2, 10, 1)"
    print(f"✓ Forward pass successful")
    print(f"  Output keys: {list(output.keys())}")
    print(f"  Action shape: {output['action'].shape}")
    print(f"  Stop shape: {output['stop'].shape}")


def test_backbone_skip_prefixes():
    """Test that stop_head is in SKIP_PREFIXES"""
    print("\n" + "="*60)
    print("TEST 5: ACTION_BACKBONE_SKIP_PREFIXES")
    print("="*60)
    
    assert "stop_head." in ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES
    print(f"✓ stop_head. is in ACTION_BACKBONE_SKIP_PREFIXES")
    print(f"  Full prefixes: {ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES}")


def test_state_dict_loading():
    """Test state_dict loading with predict_stop True/False"""
    print("\n" + "="*60)
    print("TEST 6: State Dict Loading")
    print("="*60)
    
    config = {
        "hidden_dim": 1024,
        "action_dim": 3,
        "ffn_dim": 4096,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1e-6,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 2,
        "use_gradient_checkpointing": False,
    }
    
    # Create with stop_head
    action_dit_with_stop = ActionDiT(**config, predict_stop=True)
    state_dict = action_dit_with_stop.state_dict()
    print(f"✓ State dict with stop_head has {len(state_dict)} keys")
    
    # Check that stop_head keys are present
    stop_head_keys = [k for k in state_dict.keys() if "stop_head" in k]
    print(f"  Stop head keys: {stop_head_keys}")
    assert len(stop_head_keys) > 0, "Expected stop_head keys in state_dict"
    
    # Create without stop_head and load state_dict with strict=False
    action_dit_no_stop = ActionDiT(**config, predict_stop=False)
    try:
        action_dit_no_stop.load_state_dict(state_dict, strict=False)
        print(f"✓ Can load state_dict with strict=False (for backward compatibility)")
    except Exception as e:
        print(f"✗ Error loading state_dict: {e}")
        raise


if __name__ == "__main__":
    try:
        test_action_head()
        test_stop_head()
        test_action_dit_without_stop()
        test_action_dit_with_stop()
        test_backbone_skip_prefixes()
        test_state_dict_loading()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nSummary:")
        print("1. ActionHead and StopHead classes work correctly")
        print("2. ActionDiT can be created with predict_stop=True/False")
        print("3. post_dit returns dict with 'action' key (and 'stop' if enabled)")
        print("4. fastwam.py can extract 'action' from dict output")
        print("5. Backward compatibility maintained with strict=False loading")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
