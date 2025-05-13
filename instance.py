from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz

OUT_FOLDER = 'out'

class Instance(ABC):
    def __init__(self, K):
        self.K = K

    @abstractmethod
    def next_payoffs(self):
        pass

    @abstractmethod
    def reset(self):
        pass

def plot_instance(plot_name, instance, T, max_points=100):
    instance.reset()

    payoffs = np.zeros((T, instance.K))

    for t in range(T):
        payoffs[t] = instance.next_payoffs()
    x_plt = np.linspace(0, T, min([T, max_points]), endpoint=False, dtype=int)
    plt.figure(figsize=(12, 6))
    for i in range(instance.K):
        plt.plot(x_plt+1, payoffs[x_plt, i], label=f'Arm {i+1}', alpha=0.7)

    plt.xlabel("Round ($t$)")
    plt.ylabel("Expected Reward")
    plt.grid(True, linestyle='--', alpha=0.5)
    if instance.K <= 15:
        plt.legend()
    plt.tight_layout()

    matplot2tikz.save(f'{OUT_FOLDER}/{plot_name}.tex')

    plt.show()

class RandomInstance(Instance):
    def __init__(self, K, seed):
        super().__init__(K)
        self.seed = seed

    def next_payoffs(self):
        return tuple(self.rng.uniform(0, 1) for _ in range(self.K))

    def reset(self):
        self.rng = np.random.default_rng(seed=self.seed)

class ExponentialInstance(Instance):
    def __init__(self, K, T, seed, speed=10):
        super().__init__(K)
        
        rng = np.random.default_rng(seed=seed)
        self.speed = speed
        self.c = 1 - rng.random(K)
        self.a = 1 - rng.random(K)
        self.T = T

    def next_payoffs(self):
        payoffs = self.c * (1 - np.exp(-self.a * (self.speed * self.t / self.T)))
        self.t += 1
        return payoffs

    def reset(self):
        self.t = 1

class PolynomialInstance(Instance):
    def __init__(self, K, T, seed):
        super().__init__(K)
        
        rng = np.random.default_rng(seed=seed)
        self.c = 1 - rng.random(K)
        self.a = 1 - rng.random(K)
        self.rho = 1 - rng.random(K)
        self.b = rng.exponential(K)
        self.T = T

    def next_payoffs(self):
        payoffs = self.c * (1 - self.b * ((1000 * self.t / self.T) + self.b ** ( 1. / self.rho )) ** (-self.rho) )
        self.t += 1
        return payoffs

    def reset(self):
        self.t = 1

class LineInstance(Instance):
    def __init__(self, K, T, seed):
        super().__init__(K)

        rng = np.random.default_rng(seed=seed)
        endpoints = rng.uniform(0.0, 1.0, size=(K, 2))
        self.initial = np.minimum(endpoints[:, 0], endpoints[:, 1])
        self.final = np.maximum(endpoints[:, 0], endpoints[:, 1])
        self.T = T
    
    def next_payoffs(self):
        alpha = (self.t - 1) / (self.T - 1)
        payoffs = self.initial * (1 - alpha) + self.final * alpha
        self.t += 1
        return payoffs

    def reset(self):
        self.t = 1

class StationaryInstance(Instance):
    def __init__(self, K, seed):
        super().__init__(K)

        rng = np.random.default_rng(seed=seed)
        self.payoffs = rng.uniform(low=0.0, high=1.0, size=K)
    
    def next_payoffs(self):
        return self.payoffs

    def reset(self):
        pass

class WindowedInstance(Instance):
    @abstractmethod
    def _get_window_length(self):
        pass

    def _update_window(self):
        self.d += 1
        if self.d == self._get_window_length() + 1:
            self.w += 1
            self.d = 1
            self._reset_window()
    
    def reset(self):
        self.w = 1
        self.d = 1

class LowerBoundInstance(WindowedInstance):
    def __init__(self, K, T, V_T, seed):
        super().__init__(K)
        self.K = K
        self.delta_c = 1/3
        self.m_0 = (1-2*self.delta_c) * min([1, V_T]) / 2 / T
        self.N_c = np.ceil(T**(1/5) * K**(-1/5) * min([1, V_T])**(2/5))
        self.big_delta_c = int(np.ceil(T/self.N_c))
        self.seed = seed
    
    def next_payoffs(self):
        m_high = (2 * self.N_c - (2*self.w - 2)) * self.m_0 / 2 / self.N_c
        m_mid  = (2 * self.N_c - (2*self.w - 1)) * self.m_0 / 2 / self.N_c
        m_low  = (2 * self.N_c - 2*self.w)       * self.m_0 / 2 / self.N_c
        self.payoffs[np.arange(self.K) != self.modified_arm-1] += m_mid
        self.payoffs[self.modified_arm-1] += m_high if self.d <= float(self.big_delta_c)/2 else m_low
        self._update_window()
        return self.payoffs

    def _get_window_length(self):
        return self.big_delta_c
    
    def _reset_window(self):
        self.modified_arm = self.rng.integers(1, self.K)

    def reset(self):
        super().reset()
        self.payoffs = np.full(self.K, self.delta_c)
        self.rng = np.random.default_rng(seed=self.seed)
        self._reset_window()

class FlattenedInstance(Instance):
    def __init__(self, K, instance, flattening_time):
        super().__init__(K)
        self.instance = instance
        self.flattening_time = flattening_time
    
    def next_payoffs(self):
        self.payoffs = self.instance.next_payoffs() if self.t <= self.flattening_time else self.flat_value
        if self.t == self.flattening_time:
            self.flat_value = self.payoffs
        self.t += 1
        return self.payoffs

    def reset(self):
        self.t = 1
        self.instance.reset()
