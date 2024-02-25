import torch

class STDP():
    def __init__(self, a_pos=0.0001, a_neg=-0.01, t_constant=3):
        """
        Initializes the STDP object.

        Parameters:
            a_pos (float): Positive synaptic adjustment.
            a_neg (float): Negative synaptic adjustment.
            t_constant (int): Pre- and post-synaptic time interval.
        """
        self.a_pos = a_pos
        self.a_neg = a_neg
        self.t_constant = t_constant

    def setup(self, device, n_neurons):
        """
        Hook for performing setup tasks.

        Parameters:
            device (torch.device): The torch device.
            n_neurons (int): Number of neurons in the cube.
        """
        self.device = device
        self.neurons = n_neurons

    def per_sample(self, s):
        """
        Hook for performing per sample tasks.

        Parameters:
            s (int): The sample number.
        """
        pass

    def per_time_slice(self, s, k):
        """
        Hook for performing per time-slice tasks.

        Parameters:
            s (int): The sample number.
            k (int): The time-slice number.
        """
        pass

    def train(self, aux, w_latent, spike_latent):
        """
        Learning rule as per the original NeuCube-Py implementation.

        Parameters:
            aux (torch.Tensor): Element-wise time since last spike.
            w_latent (torch.Tensor): Output weights.
            spike_latent (torch.Tensor): Output spikes.
        Returns:
            pre_updates (torch.Tensor): Pre-synaptic updates.
            pos_updates (torch.Tensor): Post-synaptic updates.
        """
        pre_w = self.a_pos*torch.exp(-aux/self.t_constant)*torch.gt(aux,0).int()
        pos_w = self.a_neg*torch.exp(-aux/self.t_constant)*torch.gt(aux,0).int()
        pre_updates = pre_w*torch.gt((w_latent.T*spike_latent).T, 0).int()
        pos_updates = pos_w*torch.gt(w_latent*spike_latent, 0).int()
        return pre_updates, pos_updates

    def reset(self):
        """
        Hook for performing reset tasks.
        """
        pass
