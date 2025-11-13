"""
訓練指標追蹤模組

追蹤訓練過程中的關鍵指標（參考 AirPilot Fig.14-16）：
- Effective Speed
- Settling Time
- Overshoot
- Mean Error
"""

import numpy as np
from typing import Dict, List, Optional
import json
import os


class TrainingMetricsTracker:
    """
    追蹤訓練過程中的關鍵指標（參考 AirPilot Fig.14-16）
    """
    
    def __init__(self, log_dir: str = "./training_metrics/"):
        """
        初始化指標追蹤器
        
        Args:
            log_dir: 日誌目錄路徑
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 指標存儲
        self.timesteps = []
        self.effective_speeds = []
        self.settling_times = []
        self.overshoots = []
        self.mean_errors = []
        
    def log_episode(self, 
                    timestep: int,
                    episode_history: Dict,
                    target_positions: Optional[List[float]] = None):
        """
        記錄一個回合的指標
        
        Args:
            timestep: 當前訓練步數
            episode_history: 回合歷史（從 env.get_history()）
            target_positions: 目標位置列表（用於計算 settling time，可選）
        """
        if not episode_history or not episode_history.get('position'):
            return
        
        positions = np.array(episode_history['position'])
        references = np.array(episode_history['reference'])
        errors = np.array(episode_history['error'])
        times = np.array(episode_history['time'])
        
        # 1. 計算平均誤差
        mean_error = np.mean(np.abs(errors))
        self.mean_errors.append(mean_error)
        
        # 2. 計算有效速度（如果適用）
        # 注意：這需要任務完成型獎勵，否則為 NaN
        effective_speed = self._calculate_effective_speed(
            positions, references, times
        )
        self.effective_speeds.append(effective_speed)
        
        # 3. 計算穩定時間
        settling_time = self._calculate_settling_time(
            errors, times, threshold=0.02  # 2% 誤差
        )
        self.settling_times.append(settling_time)
        
        # 4. 計算超調
        overshoot = self._calculate_overshoot(
            positions, references
        )
        self.overshoots.append(overshoot)
        
        # 5. 記錄時間步
        self.timesteps.append(timestep)
    
    def _calculate_effective_speed(self, positions, references, times):
        """
        計算有效速度（AirPilot Eq.9）
        
        注意：這需要任務完成型場景
        簡化實現：計算平均速度
        
        Args:
            positions: 位置數組
            references: 參考信號數組
            times: 時間數組
        
        Returns:
            有效速度（m/s），如果無法計算則返回 NaN
        """
        # 簡化實現：計算平均速度
        if len(positions) < 2:
            return np.nan
        
        distances = np.diff(positions)
        time_diffs = np.diff(times)
        
        if np.sum(time_diffs) > 0:
            avg_speed = np.sum(np.abs(distances)) / np.sum(time_diffs)
            return avg_speed
        return np.nan
    
    def _calculate_settling_time(self, errors, times, threshold=0.02):
        """
        計算穩定時間（達到 ±threshold 誤差內的時間）
        
        Args:
            errors: 誤差數組
            times: 時間數組
            threshold: 誤差閾值（相對於最大誤差的百分比）
        
        Returns:
            穩定時間（秒）
        """
        abs_errors = np.abs(errors)
        max_error = np.max(np.abs(errors)) if np.max(np.abs(errors)) > 0 else 1.0
        target_error = threshold * max_error
        
        # 找到最後一次超過閾值的時間
        above_threshold = abs_errors > target_error
        if np.any(above_threshold):
            last_above_idx = np.where(above_threshold)[0][-1]
            if last_above_idx < len(times) - 1:
                return times[last_above_idx + 1] - times[0]
        
        return times[-1] - times[0] if len(times) > 0 else np.nan  # 整個回合時間
    
    def _calculate_overshoot(self, positions, references):
        """
        計算超調量
        
        Args:
            positions: 位置數組
            references: 參考信號數組
        
        Returns:
            最大超調量
        """
        errors = positions - references
        max_overshoot = np.max(np.abs(errors))
        return max_overshoot
    
    def save(self, filename: str = "training_metrics.json"):
        """
        保存指標到 JSON
        
        Args:
            filename: 保存的檔案名稱
        """
        data = {
            'timesteps': self.timesteps,
            'effective_speeds': [float(x) if not np.isnan(x) else None for x in self.effective_speeds],
            'settling_times': [float(x) if not np.isnan(x) else None for x in self.settling_times],
            'overshoots': [float(x) if not np.isnan(x) else None for x in self.overshoots],
            'mean_errors': [float(x) for x in self.mean_errors]
        }
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✓ 訓練指標已保存到: {filepath}")
    
    def load(self, filename: str = "training_metrics.json"):
        """
        從 JSON 載入指標
        
        Args:
            filename: 載入的檔案名稱
        
        Returns:
            bool: 是否成功載入
        """
        filepath = os.path.join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.timesteps = data['timesteps']
            self.effective_speeds = [x if x is not None else np.nan for x in data['effective_speeds']]
            self.settling_times = [x if x is not None else np.nan for x in data['settling_times']]
            self.overshoots = [x if x is not None else np.nan for x in data['overshoots']]
            self.mean_errors = data['mean_errors']
            return True
        return False

