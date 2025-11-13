"""
非線性 PID 控制器（參考 AirPilot）

實現正規化速度輸出的非線性 PID 控制。
參考 AirPilot 論文 Eq.6-7：
- Eq.6: 標準 PID 計算
- Eq.7: 正規化速度輸出
"""

import numpy as np
from typing import Optional


class NonlinearPID:
    """
    非線性 PID 控制器
    
    參考 AirPilot 論文 Eq.6-7：
    - Eq.6: 標準 PID 計算
    - Eq.7: 正規化速度輸出
    """
    
    def __init__(self,
                 kp: float = 5.0,
                 ki: float = 0.1,
                 kd: float = 0.2,
                 max_velocity: float = 1.0,
                 integral_max: float = 100.0):
        """
        初始化非線性 PID 控制器
        
        Args:
            kp: 比例增益
            ki: 積分增益
            kd: 微分增益
            max_velocity: 最大速度限制
            integral_max: 積分項上限
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_velocity = max_velocity
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_max = integral_max
    
    def compute(self, error: float, dt: float) -> float:
        """
        計算非線性 PID 控制輸出（Eq.6-7）
        
        Args:
            error: 當前誤差
            dt: 時間步長
        
        Returns:
            normalized_velocity: 正規化速度指令
        """
        # 更新積分項（含 anti-windup）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # 計算微分項
        if dt > 0:
            error_dot = (error - self.last_error) / dt
        else:
            error_dot = 0.0
        
        # PID 控制律（Eq.6）
        velocity = self.kp * error + self.ki * self.integral + self.kd * error_dot
        
        # 正規化速度（Eq.7）
        normalized_velocity = velocity / (np.abs(velocity) + 1.0)
        normalized_velocity = np.clip(
            normalized_velocity,
            -self.max_velocity,
            self.max_velocity
        )
        
        self.last_error = error
        return normalized_velocity
    
    def update_gains(self, kp: float, ki: float, kd: float):
        """
        更新 PID 增益（由 RL 智能體調用）
        
        Args:
            kp: 新的比例增益
            ki: 新的積分增益
            kd: 新的微分增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def reset(self):
        """
        重置控制器狀態（用於新 episode）
        """
        self.integral = 0.0
        self.last_error = 0.0
    
    def get_state(self) -> dict:
        """
        獲取當前控制器狀態（用於調試/可視化）
        
        Returns:
            包含增益和積分項的字典
        """
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral': self.integral,
            'last_error': self.last_error
        }

