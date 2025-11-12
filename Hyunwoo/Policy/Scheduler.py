import math
from typing import List, Tuple, Dict, Optional

class BaseScheduler:
    """모든 스케줄러의 베이스. __call__(t) == value(t)."""
    def value(self, t: float) -> float:
        raise NotImplementedError
    
    def __call__(self, t: float) -> float:
        return self.value(t)
    
    def get_config(self) -> Dict:
        raise NotImplementedError


# ---------------------------
# 단일 구간형 스케줄러들
# ---------------------------

class ConstantScheduler(BaseScheduler):
    def __init__(self, v: float):
        self.v = float(v)
    def value(self, t: float) -> float:
        return self.v
    def get_config(self):
        return {"type": "constant", "v": self.v}


class LinearSchedulerInt(BaseScheduler):
    """t0~t1 구간에서 v0->v1 선형 보간."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return int(self.v0)
            if t >= self.t1: return int(self.v1)
        x = (t - self.t0) / self.dt
        return int(self.v0 + (self.v1 - self.v0) * x)
    def get_config(self):
        return {"type": "linear", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}

class LinearScheduler(BaseScheduler):
    """t0~t1 구간에서 v0->v1 선형 보간."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.v0
            if t >= self.t1: return self.v1
        x = (t - self.t0) / self.dt
        return self.v0 + (self.v1 - self.v0) * x
    def get_config(self):
        return {"type": "linear", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}
    

class ExponentialScheduler(BaseScheduler):
    """t0~t1 구간에서 v0 -> v1 지수 보간 (v>0 권장)."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        assert v0 != 0 and v1 != 0, "v0, v1 must be non-zero"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
        self.r = self.v1 / self.v0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.v0
            if t >= self.t1: return self.v1
        x = (t - self.t0) / self.dt
        return self.v0 * (self.r ** x)
    def get_config(self):
        return {"type": "exponential", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}


class ExponentialSchedulerInt(BaseScheduler):
    """t0~t1 구간에서 v0 -> v1 지수 보간 (v>0 권장)."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        assert v0 != 0 and v1 != 0, "v0, v1 must be non-zero"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
        self.r = self.v1 / self.v0

    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return int(self.v0)
            if t >= self.t1: return int(self.v1)
        x = (t - self.t0) / self.dt
        return int(self.v0 * (self.r ** x))
    def get_config(self):
        return {"type": "exponential", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}

class PolynomialScheduler(BaseScheduler):
    """다항 보간: v(t)=v0 + (v1-v0)*((t-t0)/(t1-t0))**power"""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, power: float = 2.0, clip: bool = True):
        assert t1 > t0
        self.t0, self.v0, self.t1, self.v1 = float(t0), float(v0), float(t1), float(v1)
        self.power, self.clip = float(power), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.v0
            if t >= self.t1: return self.v1
        x = (t - self.t0) / self.dt
        return self.v0 + (self.v1 - self.v0) * (x ** self.power)
    def get_config(self):
        return {"type": "polynomial", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "power": self.power, "clip": self.clip}
    


class CosineScheduler(BaseScheduler):
    """코사인 애닐링: v = v_min + 0.5*(v_max - v_min)*(1 + cos(pi * progress + phase))"""
    def __init__(self, t0: float, t1: float, v_min: float, v_max: float, phase: float = 0.0, clip: bool = True):
        assert t1 > t0
        self.t0, self.t1, self.vmin, self.vmax, self.phase, self.clip = float(t0), float(t1), float(v_min), float(v_max), float(phase), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.vmax
            if t >= self.t1: return self.vmin
        x = (t - self.t0) / self.dt
        return self.vmin + 0.5 * (self.vmax - self.vmin) * (1.0 + math.cos(math.pi * x + self.phase))
    def get_config(self):
        return {"type": "cosine", "t0": self.t0, "t1": self.t1, "v_min": self.vmin, "v_max": self.vmax, "phase": self.phase, "clip": self.clip}


class SigmoidScheduler(BaseScheduler):
    """시그모이드 전이: 중앙 t_mid, 기울기 k."""
    def __init__(self, v_low: float, v_high: float, t_mid: float, k: float = 1.0):
        self.vl, self.vh, self.t_mid, self.k = float(v_low), float(v_high), float(t_mid), float(k)
    def value(self, t: float) -> float:
        s = 1.0 / (1.0 + math.exp(-self.k * (t - self.t_mid)))
        return self.vl + (self.vh - self.vl) * s
    def get_config(self):
        return {"type": "sigmoid", "v_low": self.vl, "v_high": self.vh, "t_mid": self.t_mid, "k": self.k}


class InverseSqrtScheduler(BaseScheduler):
    """v = scale / sqrt(t + offset). 보통 옵티마이저 LR에 사용."""
    def __init__(self, scale: float, offset: float = 1.0):
        assert offset >= 0
        self.scale, self.offset = float(scale), float(offset)
    def value(self, t: float) -> float:
        return self.scale / math.sqrt(max(t + self.offset, 1e-12))
    def get_config(self):
        return {"type": "inverse_sqrt", "scale": self.scale, "offset": self.offset}


# ---------------------------
# 복합/구간/주기형 스케줄러들
# ---------------------------

class StepScheduler(BaseScheduler):
    """계단형: [(t1, v1), (t2, v2), ...], t < t1 -> v1, t1<=t<t2 -> v2 ... 마지막 이상은 마지막 값."""
    def __init__(self, steps: List[Tuple[float, float]]):
        assert len(steps) > 0
        self.steps = sorted((float(t), float(v)) for t, v in steps)
    def value(self, t: float) -> float:
        for tt, vv in self.steps:
            if t < tt:
                return vv
        return self.steps[-1][1]
    def get_config(self):
        return {"type": "step", "steps": self.steps}


class PiecewiseScheduler(BaseScheduler):
    """
    구간별 서로 다른 스케줄러 연결.
    segments: List of (t_start, t_end, scheduler)
    t < 첫 구간 -> 첫 구간 시작값, t > 마지막 구간 -> 마지막 구간 끝값
    """
    def __init__(self, segments: List[Tuple[float, float, BaseScheduler]]):
        assert len(segments) > 0
        self.segments = [(float(a), float(b), s) for a, b, s in segments]
        for a, b, s in self.segments:
            assert b > a
    def value(self, t: float) -> float:
        # 앞/뒤는 고정
        a0, b0, s0 = self.segments[0]
        an, bn, sn = self.segments[-1]
        if t <= a0:
            return s0.value(a0)
        if t >= bn:
            return sn.value(bn)
        for a, b, s in self.segments:
            if a <= t <= b:
                return s.value(t)
        # 이론상 도달X
        return sn.value(bn)
    def get_config(self):
        return {
            "type": "piecewise",
            "segments": [
                {"t_start": a, "t_end": b, "scheduler": s.get_config()} for a, b, s in self.segments
            ],
        }


class WarmupHoldDecayScheduler(BaseScheduler):
    """
    Linear warmup -> (optional) hold -> cosine decay
    예: LR warmup 후 코사인 감쇠
    """
    def __init__(self, warmup_end: float, hold_end: float, total_end: float,
                 v_warmup_start: float, v_warmup_end: float,
                 v_final: float):
        assert 0 <= warmup_end <= hold_end <= total_end
        self.warm = LinearScheduler(0.0, v_warmup_start, warmup_end, v_warmup_end, clip=True)
        self.hold_end = hold_end
        self.total_end = total_end
        self.v_hold = v_warmup_end
        self.cos = CosineScheduler(hold_end, total_end, v_min=v_final, v_max=v_warmup_end, clip=True)
    def value(self, t: float) -> float:
        if t <= self.warm.t1:
            return self.warm.value(t)
        if t <= self.hold_end:
            return self.v_hold
        if t <= self.total_end:
            return self.cos.value(t)
        return self.cos.value(self.total_end)
    def get_config(self):
        return {
            "type": "warmup_hold_decay",
            "warmup": self.warm.get_config(),
            "hold_end": self.hold_end,
            "total_end": self.total_end,
            "cosine": self.cos.get_config(),
        }


class CyclicalScheduler(BaseScheduler):
    """
    Triangular (sawtooth) 주기 스케줄.
    period: 주기 길이
    step_ratio: 상승 비율 (0~1), 기본 0.5 -> 절반 상승/절반 하강
    """
    def __init__(self, v_min: float, v_max: float, period: float, step_ratio: float = 0.5, start_t: float = 0.0):
        assert v_max >= v_min
        assert period > 0
        assert 0.0 < step_ratio < 1.0
        self.vmin, self.vmax = float(v_min), float(v_max)
        self.period, self.step_ratio, self.start_t = float(period), float(step_ratio), float(start_t)
    def value(self, t: float) -> float:
        x = (t - self.start_t) % self.period
        up_T = self.period * self.step_ratio
        if x <= up_T:
            # 상승: vmin -> vmax
            return self.vmin + (self.vmax - self.vmin) * (x / up_T)
        else:
            # 하강: vmax -> vmin
            d = (x - up_T) / (self.period - up_T)
            return self.vmax - (self.vmax - self.vmin) * d
    def get_config(self):
        return {"type": "cyclical", "v_min": self.vmin, "v_max": self.vmax, "period": self.period, "step_ratio": self.step_ratio, "start_t": self.start_t}


class CosineRestartScheduler(BaseScheduler):
    """
    SGDR 스타일 코사인 리스타트.
    T0: 첫 주기 길이
    T_mult: 매 주기 길이 배수 (예: 2면 50, 100, 200 ...)
    """
    def __init__(self, v_min: float, v_max: float, T0: int, T_mult: float = 2.0, start_t: int = 0):
        assert T0 > 0 and T_mult >= 1.0
        self.vmin, self.vmax = float(v_min), float(v_max)
        self.T0, self.T_mult, self.start_t = int(T0), float(T_mult), int(start_t)
    def value(self, t: int) -> float:
        # 정수 스텝 기준
        step = max(int(t) - self.start_t, 0)
        Ti = self.T0
        acc = 0
        while step >= Ti:
            step -= Ti
            acc += Ti
            Ti = int(Ti * self.T_mult)
        # 현재 주기에서의 코사인
        if Ti <= 0: Ti = 1
        progress = step / Ti
        return self.vmin + 0.5 * (self.vmax - self.vmin) * (1 + math.cos(math.pi * progress))
    def get_config(self):
        return {"type": "cosine_restart", "v_min": self.vmin, "v_max": self.vmax, "T0": self.T0, "T_mult": self.T_mult, "start_t": self.start_t}


class BooleanScheduler(BaseScheduler):
    """
    주어진 시간 t가 지정된 구간 안에 있으면 True, 아니면 False를 반환하는 스케줄러.
    
    Args:
        true_intervals (List[Tuple[float, float]]): True로 반환할 (시작, 끝) 구간 리스트.
        default (bool): 기본값 (어느 구간에도 속하지 않을 때 반환할 값)
        inclusive (bool): 구간의 양 끝 포함 여부 (기본값 True)
    """
    def __init__(self,
                 true_intervals: List[Tuple[float, float]],
                 default: bool = False,
                 inclusive: bool = True):
        self.true_intervals = true_intervals
        self.default = default
        self.inclusive = inclusive

    def value(self, t: float) -> bool:
        for (start, end) in self.true_intervals:
            if self.inclusive:
                if start <= t <= end:
                    return True
            else:
                if start < t < end:
                    return True
        return self.default

    def get_config(self) -> Dict:
        return {
            "type": "boolean",
            "true_intervals": self.true_intervals,
            "default": self.default,
            "inclusive": self.inclusive
        }

