import torch

class NRDP():
    def __init__(self, min_a=0.0, max_a=0.901679611594126, gain_a=0.5428717518563672,
                 min_n=0.0, max_n=0.23001290732040292, gain_n=0.011660312977761912,
                 min_ga=0.0, max_ga=0.7554145024515596, gain_ga=0.3859076787035615,
                 min_gb=0.0, max_gb=0.7954714253083993, gain_gb=0.11032115434326673,
                 time_window=10, gaba_impact=0.01, gaba_rate=0.7):
        """
        Initializes the NRDP object. This implementation is highly sensitive to values
        supplied for minimum/maximum Neuroreceptor levels and gaining rates.

        Parameters:
            min_a (float): Minimum level for AMPAR
            max_a (float): Maximum level for AMPAR
            gain_a (float): Gaining rate for AMPAR
            min_n (float): Minimum level for NMDAR
            max_n (float): Maximum level for NMDAR
            gain_n (float): Gaining rate for NMDAR
            min_ga (float): Minimum level for GABAa
            max_ga (float): Maximum level for GABAa
            gain_ga (float): Gaining rate for GABAa
            min_gb (float): Minimum level for GABAb
            max_gb (float): Maximum level for GABAb
            gain_gb (float): Gaining rate for GABAb
            time_window (int): Time window for AMPAR
            gaba_impact (float): Rate that the GABA receptors impact AMPAR
            gaba_rate (float): GABAa activation probability
        """
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

    def setup(self, device, n_neurons):
        """
        Hook for performing setup tasks.

        Parameters:
            device (torch.device): The torch device.
            n_neurons (int): Number of neurons in the cube.
        """
        self.device = device
        self.n_neurons = n_neurons

    def per_sample(self, s):
        """
        Hook for performing per sample tasks.

        Parameters:
            s (int): The sample number.
        """
        self.firing_state = torch.zeros(self.n_neurons).to(self.device) # Element-wise count since last spike
        self.a_state = torch.zeros(self.n_neurons).to(self.device) # Element-wise AMPAR state
        self.a_t1 = torch.zeros(self.n_neurons).to(self.device) # A(t-1)
        self.n_t1 = torch.zeros(self.n_neurons).to(self.device) # N(t-1)
        self.g_t1 = torch.zeros(self.n_neurons).to(self.device) # G(t-1)

    def per_time_slice(self, s, k):
        """
        Hook for performing per time-slice tasks.

        Parameters:
            s (int): The sample number.
            k (int): The time-slice number.
        """
        self.gaba_gains = torch.where(torch.rand((self.n_neurons,)) < self.gaba_rate, self.gain_ga, self.gain_gb)
        self.gaba_mins = torch.where(torch.rand((self.n_neurons,)) < self.gaba_rate, self.min_ga, self.min_gb)
        self.gaba_maxes = torch.where(torch.rand((self.n_neurons,)) < self.gaba_rate, self.max_ga, self.max_gb)

    def train(self, aux, w_latent, spike_latent):
        """
        Implements the NRDP learning rule as per: https://ieeexplore.ieee.org/document/8118188.

        Parameters:
            aux (torch.Tensor): Element-wise time since last spike.
            w_latent (torch.Tensor): Output weights.
            spike_latent (torch.Tensor): Output spikes.
        Returns:
            pre_updates (torch.Tensor): Pre-synaptic updates.
            pos_updates (torch.Tensor): Post-synaptic updates.
        """
        # calculate A(t) when Sij = 1
        a_t = torch.where(spike_latent > 0,
                          torch.minimum(torch.full_like(self.a_t1, self.max_a), self.a_t1+self.gain_a),
                          # calculate A(t) when Sij = 0 for the length of time window
                          torch.where(self.firing_state >= self.time_window,
                                      torch.maximum(torch.full_like(self.a_t1, self.min_a), self.a_t1-(self.gaba_impact*self.gaba_gains)),
                                      self.a_t1))
        self.firing_state[spike_latent < 1] += 1
        # hold A(t) at maximum level for duration of time window
        a_t = torch.where(a_t >= self.max_a,
                          torch.where(self.a_state <= self.time_window, self.max_a, self.a_t1),
                          self.a_t1)
        self.a_state[a_t >= self.max_a] += 1
        # calculate N(t) when A(t) reaches maximum level and Sij = 1
        n_t = torch.where(a_t >= self.max_a,
                          torch.where(spike_latent > 0,
                                      torch.minimum(torch.full_like(self.n_t1, self.max_n), self.n_t1+self.gain_n),
                                      torch.maximum(torch.full_like(self.n_t1, self.min_n), self.n_t1-self.gain_n)),
                          self.min_n) # otherwise NMDAR is not activated
        # calculate G(t) when Sij = 1
        g_t = torch.where(spike_latent > 0,
                          torch.maximum(self.gaba_mins, self.g_t1-self.gaba_gains),
                          torch.minimum(self.gaba_maxes, self.g_t1+(self.gaba_gains*aux)))

        self.a_t1 = a_t
        self.n_t1 = n_t
        self.g_t1 = g_t

        pre_updates = torch.zeros(self.n_neurons, self.n_neurons)
        pos_updates = ((n_t + a_t) - g_t)*torch.gt(w_latent*spike_latent, 0).int()

        return pre_updates, pos_updates

    def reset(self):
        """
        Hook for performing reset tasks.
        """
        self.firing_state[self.firing_state > self.time_window] = 0
        self.a_state[self.a_state > self.time_window] = 0