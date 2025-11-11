"""
快速檢查訓練設備的腳本

運行此腳本以確認 PyTorch 和 Stable-Baselines3 使用的設備。
"""

import torch
import sys

def check_device():
    """檢查可用的設備和配置"""
    print("=" * 60)
    print("設備檢查")
    print("=" * 60)
    
    # PyTorch 資訊
    print("\n[1] PyTorch 資訊:")
    print(f"  版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"  GPU 數量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}:")
            print(f"    名稱: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    記憶體: {props.total_memory / 1e9:.2f} GB")
            print(f"    計算能力: {props.major}.{props.minor}")
    else:
        print("  ⚠️  未檢測到 CUDA，將使用 CPU")
    
    # 測試設備選擇
    print("\n[2] 設備選擇邏輯:")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  自動選擇: {device}")
        print(f"  設備名稱: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"  自動選擇: {device}")
    
    # Stable-Baselines3 檢查
    print("\n[3] Stable-Baselines3 檢查:")
    try:
        from stable_baselines3 import PPO
        print("  ✓ Stable-Baselines3 已安裝")
        
        # 檢查默認設備行為
        print("\n  默認行為:")
        print("    - 如果未指定 device 參數，SB3 會自動檢測")
        print("    - 有 CUDA → 使用 GPU")
        print("    - 無 CUDA → 使用 CPU")
        
    except ImportError:
        print("  ✗ Stable-Baselines3 未安裝")
        print("    安裝: pip install stable-baselines3")
    
    # 建議
    print("\n[4] 建議:")
    if torch.cuda.is_available():
        print("  ✓ 檢測到 GPU，訓練將自動使用 GPU")
        print("  ✓ 如果希望強制使用 CPU，可在 train.py 中指定 device='cpu'")
    else:
        print("  ℹ️  未檢測到 GPU，訓練將使用 CPU")
        print("  ℹ️  對於本專案（小網路），CPU 訓練通常足夠快")
    
    print("\n" + "=" * 60)
    print("檢查完成")
    print("=" * 60)
    
    return device

if __name__ == "__main__":
    device = check_device()
    sys.exit(0)

