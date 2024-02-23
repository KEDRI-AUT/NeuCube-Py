import torch
from numpy.random import uniform as unif

# NRDP implementation as per: https://ieeexplore.ieee.org/document/8118188
class NRDP():
    def __init__(self, device, n_neurons, min_a=0.0, max_a=0.901679611594126, gain_a=0.5428717518563672,
                 min_n=0.0, max_n=0.23001290732040292, gain_n=0.011660312977761912,
                 min_ga=0.0, max_ga=0.7554145024515596, gain_ga=0.3859076787035615,
                 min_gb=0.0, max_gb=0.7954714253083993, gain_gb=0.11032115434326673,
                 time_window=10, gaba_impact=0.01, gaba_rate=0.7):
        self.device = device
        self.n_neurons = n_neurons
        self.min_a = min_a
        self.max_a = max_a
        self.gain_a = gain_a
        self.min_n = min_n
        self.max_n = max_n
        self.gain_n = gain_n
        self.min_ga = min_ga
        self.max_ga = max_ga
        self.gain_ga = gain_ga
        self.min_gb = min_gb
        self.max_gb = max_gb
        self.gain_gb = gain_gb
        self.time_window = time_window
        self.gaba_impact = gaba_impact
        self.gaba_rate = gaba_rate

    def _is_gaba_activated(self):
        return unif(0, 1) < self.gaba_rate

    def _get_gaba_gain(self):
        return self.gain_ga if self._is_gaba_activated() else self.gain_gb

    def _get_gaba_min(self):
        return self.min_ga if self._is_gaba_activated() else self.min_gb

    def _get_gaba_max(self):
        return self.max_ga if self._is_gaba_activated() else self.max_gb

    def per_sample(self, s):
        self.firing_state = torch.zeros(self.n_neurons).to(self.device)
        self.a_state = torch.zeros(self.n_neurons).to(self.device)
        self.a_t = torch.zeros(self.n_neurons).to(self.device)
        self.n_t = torch.zeros(self.n_neurons).to(self.device)
        self.g_t = torch.zeros(self.n_neurons).to(self.device)

    def per_time_slice(self, s, k):
        pass

    def train(self, aux, _, spike_latent):
        a_t = torch.where(self.firing_state >= self.time_window,
                          torch.maximum(torch.full_like(self.a_t, self.min_a), self.a_t-(self.gaba_impact*self._get_gaba_gain())),
                          torch.minimum(torch.full_like(self.a_t, self.max_a), self.a_t+self.gain_a))
        self.firing_state[spike_latent < 1] += 1
        # TODO: set A(t) to MAX_A for duration of time window
        a_t = torch.where(a_t >= self.max_a,
                          torch.where(self.a_state < self.time_window, self.max_a, a_t),
                          a_t)
        self.a_state[a_t >= self.max_a] += 1
        # only calculate N(t) when A(t) reaches its max level and there is a spike
        n_t = torch.where(a_t >= self.max_a,
                          torch.where(spike_latent > 0,
                                      torch.maximum(torch.full_like(self.n_t, self.min_n), self.n_t-self.gain_n),
                                      torch.minimum(torch.full_like(self.n_t, self.max_n), self.n_t+self.gain_n)),
                          self.min_n)
        g_t = torch.where(spike_latent > 0,
                          torch.maximum(torch.full_like(self.g_t, self._get_gaba_min()), self.g_t-self._get_gaba_gain()),
                          torch.minimum(torch.full_like(self.g_t, self._get_gaba_max()), self.g_t+self._get_gaba_gain()*aux))

        return torch.zeros(self.n_neurons, self.n_neurons), (n_t + a_t) - g_t  # Wij

    def reset(self):
        self.firing_state[self.firing_state >= self.time_window] = 0
        self.a_state[self.a_state >= self.time_window] = 0
