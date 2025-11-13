"""
標準線性 PID 控制器

從 dppo_pid_env.py 提取的 PID 邏輯，用於模組化設計。
"""

import numpy as np
from typing import Optional


class LinearPID:
    """
    標準線性 PID 控制器
    
    實現標準並聯式 PID 控制律：
    u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
    """
    
    def __init__(self, 
                 kp: float = 5.0,
                 ki: float = 0.1,
                 kd: float = 0.2,
                 integral_max: float = 100.0):
        """
        初始化 PID 控制器
        
        Args:
            kp: 比例增益
            ki: 積分增益
            kd: 微分增益
            integral_max: 積分項上限（anti-windup）
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_max = integral_max
    
    def compute(self, error: float, dt: float) -> float:
        """
        計算 PID 控制輸出
        
        Args:
            error: 當前誤差 e(t) = r(t) - x(t)
            dt: 時間步長
        
        Returns:
            control_output: 控制輸入 u(t)
        """
        # 更新積分項（含 anti-windup）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        
        # 計算微分項
        if dt > 0:
            error_dot = (error - self.last_error) / dt
        else:
            error_dot = 0.0
        
        # PID 控制律
        u = self.kp * error + self.ki * self.integral + self.kd * error_dot
        
        # 更新歷史
        self.last_error = error
        
        return u
    
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

