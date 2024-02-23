import torch

class STDP():
    def __init__(self, device, n_neurons, tau_pos=0.0001, tau_neg=-0.01, t_constant=3):
        self.device = device
        self.n_neurons = n_neurons
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.t_constant = t_constant

    def per_sample(self, s):
        pass

    def per_time_slice(self, s, k):
        pass

    def train(self, aux, w_latent, spike_latent):
        pre_w = self.tau_pos*torch.exp(-aux/self.t_constant)*torch.gt(aux,0).int()
        pos_w = self.tau_neg*torch.exp(-aux/self.t_constant)*torch.gt(aux,0).int()
        pre_updates = pre_w*torch.gt((w_latent.T*spike_latent).T, 0).int()
        pos_updates = pos_w*torch.gt(w_latent*spike_latent, 0).int()
        return pre_updates, pos_updates

    def reset(self):
        pass
