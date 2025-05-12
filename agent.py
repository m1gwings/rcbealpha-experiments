from abc import ABC, abstractmethod
import numpy as np
import matplot2tikz

from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, K, name):
        self.K = K
        self.name = name

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def observe(self, reward):
        pass

    @abstractmethod
    def reset(self):
        pass

class RandomAgent(Agent):
    def __init__(self, K, seed):
        super().__init__(K, 'Random')
        self.seed = seed

    def pull(self):
        return self.rng.integers(1, self.K)

    def observe(self, reward):
        pass
    
    def reset(self):
        self.rng = np.random.default_rng(seed=self.seed)

class WindowedAgent(Agent):
    def __init__(self, K, name):
        super().__init__(K, name)

    @abstractmethod
    def _get_window_length(self):
        pass

    @abstractmethod
    def _reset_window(self):
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

class RCBEAlphaAgent(WindowedAgent):
    def __init__(self, K, alpha, sigma):
        super().__init__(K, '$\texttt{RC-BE}(\\alpha)$')
        self.alpha = alpha
        self.sigma = sigma

    def pull(self):
        self.last_pulled = min(self.B) + 1
        return self.last_pulled

    def observe(self, reward):
        self.S[self.last_pulled - 1] += reward
        self.B.remove(self.last_pulled - 1)

        if not self.B:
            self._eliminate_arms()

        self._update_window()

    def _get_window_length(self):
        return int(np.ceil(self.w ** self.alpha))

    def _eliminate_arms(self):
        S_star = max([self.S[i] for i in self.A])
        delta_w = self._get_window_length()
        B_w = 2 * (1 + 2 * self.sigma * np.sqrt(delta_w * np.log(2 * self.K * delta_w)))
        self.A = {i for i in self.A if self.S[i] + B_w >= S_star}
        self.B = self.A.copy()

    def _reset_window(self):
        self.S = [0.0] * self.K
        self.A = set(range(self.K))
        self.B = self.A.copy()
    
    def reset(self):
        super().reset()
        self._reset_window()

class Rexp3Agent(WindowedAgent):
    def __init__(self, K, delta, gamma, seed):
        super().__init__(K, '$\\texttt{Rexp3}$')
        self.delta = delta
        self.gamma = gamma
        self.seed = seed

    def pull(self):
        self.last_pulled = self.rng.choice(np.arange(1, self.K + 1), p=self.p)
        return self.last_pulled

    def observe(self, reward):
        self.weights[self.last_pulled - 1] *= np.exp(self.gamma / self.K * reward / self.p[self.last_pulled - 1])
        self._update_p()
        self._update_window()

    def _get_window_length(self):
        return self.delta

    def _reset_window(self):
        self.weights = np.ones(self.K)
        self._update_p()

    def _update_p(self):
        self.p = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.K
    
    @staticmethod
    def tune_delta(T, K, V_T):
        return int(np.ceil( (K * np.log(K)) ** (1/3) * (T/V_T) ** (2/3) ))

    @staticmethod
    def tune_gamma(T, K, V_T):
        return min([1, np.sqrt(K * np.log(K) / (np.e - 1) / Rexp3Agent.tune_delta(T, K, V_T))])

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng(seed=self.seed)
        self._reset_window()

class RLessUCBAgent(Agent):
    def __init__(self, K, epsilon, alpha, sigma, initial_capacity=2**20):
        super().__init__(K, '$\texttt{R-less-UCB}$')
        self.epsilon = epsilon
        self.alpha = alpha
        self.sigma = sigma
        self.rewards = np.full((self.K, initial_capacity), np.nan)

    def pull(self):
        self.last_pulled = np.argmax(self.ucbs) + 1
        return self.last_pulled

    def observe(self, reward):
        if self.rewards_counts[self.last_pulled - 1] >= self.rewards.shape[1]:
            self._expand_rewards()

        self.rewards[self.last_pulled - 1, self.rewards_counts[self.last_pulled - 1]] = reward
        self.rewards_counts[self.last_pulled - 1] += 1
        
        old_h = self.h[self.last_pulled-1]
        self.h[self.last_pulled-1] = int(np.floor(self.epsilon * self.rewards_counts[self.last_pulled - 1]))
        h = self.h[self.last_pulled-1]
        n = self.rewards_counts[self.last_pulled-1]
        old_reward = self.rewards[self.last_pulled-1, n-h-1]
        old_oldish_reward = self.rewards[self.last_pulled-1, n-2*h]
        old_old_reward = self.rewards[self.last_pulled-1, n-2*h-1]
        if h == old_h:
            self.a[self.last_pulled-1] += reward - old_reward
            self.b[self.last_pulled-1] += old_reward - old_old_reward
            self.c[self.last_pulled-1] += n*reward - (n-h)*old_reward
            self.d[self.last_pulled-1] += n*old_reward - (n-h)*old_old_reward
        else:
            self.a[self.last_pulled-1] += reward
            self.b[self.last_pulled-1] += old_oldish_reward
            self.c[self.last_pulled-1] += n*reward
            self.d[self.last_pulled-1] += (n-h)*old_oldish_reward + self.b[self.last_pulled-1]
        self.t += 1

        self._update_preds_ucbs()

    def _expand_rewards(self):
        new_capacity = self.rewards.shape[1] * 2
        new_rewards = np.full((self.K, new_capacity), np.nan)
        new_rewards[:, :self.rewards.shape[1]] = self.rewards
        self.rewards = new_rewards

    def _update_preds_ucbs(self):
        self.preds[self.h == 0] = np.inf
        self.ucbs[self.h == 0] = np.inf

        self.preds[self.h > 0] = 1/self.h[self.h > 0] * (self.a[self.h > 0] + self.t*(self.a[self.h > 0] - self.b[self.h > 0])/self.h[self.h > 0] - (self.c[self.h > 0] - self.d[self.h > 0])/self.h[self.h > 0])

        delta_t = self.t ** (-self.alpha)
        self.ucbs[self.h > 0] = self.preds[self.h > 0] + self.sigma * (self.t - self.rewards_counts[self.h > 0] + self.h[self.h > 0] - 1) * np.sqrt(10 * np.log(1/delta_t) / (self.h[self.h > 0] ** 3))
    
    def reset(self):
        self.rewards_counts = np.zeros(self.K, dtype=int)
        self.h = np.zeros(self.K, dtype=int)
        self.preds = np.full(self.K, np.inf)
        self.ucbs = np.full(self.K, np.inf)
        self.a = np.zeros(self.K)
        self.b = np.zeros(self.K)
        self.c = np.zeros(self.K)
        self.d = np.zeros(self.K)
        self.t = 1

class UCB1Agent(Agent):
    def __init__(self, K, sigma):
        super().__init__(K, '$\texttt{UCB1}$')
        self.sigma = sigma

    def pull(self):
        self.last_pulled = np.argmax(self.ucbs) + 1
        return self.last_pulled

    def observe(self, reward):
        self.counts[self.last_pulled - 1] += 1
        self.cumulative_rewards[self.last_pulled - 1] += reward

        self.t += 1
        self._update_ucbs()
    
    def _update_ucbs(self):
        self.ucbs[self.counts == 0] = np.inf
        self.ucbs[self.counts > 0] = self.cumulative_rewards[self.counts > 0] / self.counts[self.counts > 0] + self.sigma * np.sqrt(4 * np.log(self.t) / self.counts[self.counts > 0])
    
    def reset(self):
        self.counts = np.zeros(self.K, dtype=int)
        self.cumulative_rewards = np.zeros(self.K)
        self.ucbs = np.full(self.K, np.inf)
        self.t = 1
