from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DriftStatus:
    js_divergence: float
    is_drifting: bool
    cache_mean_reward: float
    fresh_mean_reward: float
    cache_samples: int
    fresh_samples: int
    recommendation: str


class DriftDetector:
    _MIN_SAMPLES = 10

    def __init__(self, window_size: int = 50, threshold: float = 0.15):
        self.window_size = window_size
        self.threshold = threshold
        self.cache_rewards: deque[float] = deque(maxlen=window_size)
        self.fresh_rewards: deque[float] = deque(maxlen=window_size)

    def record(self, source: str, reward: float) -> None:
        if source == "cache":
            self.cache_rewards.append(reward)
        else:
            self.fresh_rewards.append(reward)

    def check_drift(self) -> DriftStatus:
        cache_arr = np.array(self.cache_rewards)
        fresh_arr = np.array(self.fresh_rewards)

        cache_mean = float(cache_arr.mean()) if len(cache_arr) > 0 else 0.0
        fresh_mean = float(fresh_arr.mean()) if len(fresh_arr) > 0 else 0.0

        if len(cache_arr) < self._MIN_SAMPLES or len(fresh_arr) < self._MIN_SAMPLES:
            return DriftStatus(
                js_divergence=0.0,
                is_drifting=False,
                cache_mean_reward=cache_mean,
                fresh_mean_reward=fresh_mean,
                cache_samples=len(cache_arr),
                fresh_samples=len(fresh_arr),
                recommendation="INSUFFICIENT_DATA",
            )

        js_div = self._js_divergence(cache_arr, fresh_arr)
        is_drifting = js_div > self.threshold and cache_mean < fresh_mean

        return DriftStatus(
            js_divergence=js_div,
            is_drifting=is_drifting,
            cache_mean_reward=cache_mean,
            fresh_mean_reward=fresh_mean,
            cache_samples=len(cache_arr),
            fresh_samples=len(fresh_arr),
            recommendation=(
                "CACHE_DEGRADING_EVICT_LOW_QUALITY" if is_drifting else "OK"
            ),
        )

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        combined = np.concatenate([p, q])
        lo, hi = float(combined.min()), float(combined.max())

        if lo == hi:
            return 0.0

        bins = np.linspace(lo, hi, 21)

        p_hist = np.histogram(p, bins=bins)[0].astype(float) + 1e-10
        q_hist = np.histogram(q, bins=bins)[0].astype(float) + 1e-10

        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        m = 0.5 * (p_hist + q_hist)

        kl_pm = float(np.sum(p_hist * np.log(p_hist / m)))
        kl_qm = float(np.sum(q_hist * np.log(q_hist / m)))

        return 0.5 * kl_pm + 0.5 * kl_qm

    def reset(self) -> None:
        self.cache_rewards.clear()
        self.fresh_rewards.clear()

    @property
    def cache_count(self) -> int:
        return len(self.cache_rewards)

    @property
    def fresh_count(self) -> int:
        return len(self.fresh_rewards)
