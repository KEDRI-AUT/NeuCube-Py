import torch
from abc import ABC, abstractmethod

class Sampler(ABC):
    """
    Abstract base class for sampling state vectors for classification.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, X_in):
        """
        Abstract method to sample from input data.

        Parameters:
            X_in: Input data.

        Returns:
            State vectors.
        """
        pass
        

class SpikeCount(Sampler):
    """
    Sampler that calculates the spike count from spike activity.
    """

    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        """
        Calculates the spike count from spike activity.

        Parameters:
            spike_activity: Spike activity tensor or array-like object.

        Returns:
            State vectors (spike count for each sample).
        """
        return spike_activity.sum(axis=1)

class MeanFiringRate(Sampler):
    """
    Sampler that calculates the mean firing rate from spike activity.
    """

    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        """
        Calculates the mean firing rate from spike activity.

        Parameters:
            spike_activity: Spike activity tensor or array-like object.

        Returns:
            State vectors (mean firing rate for each sample).
        """
        return spike_activity.mean(axis=1)

import torch

class Binning(Sampler):
    """
    Sampler that calculates the temporal binning from spike activity.
    """

    def __init__(self, bin_size=10):
        """
        Initializes the TemporalBinning sampler.

        Parameters:
            bin_size (int): Size of the bin (default: 10).
        """
        self.bin_size = bin_size

    def sample(self, spike_activity):
        """
        Calculates the temporal binning from spike activity.

        Parameters:
            spike_activity (torch.Tensor): Spike activity tensor or array-like object.

        Returns:
            torch.Tensor: State vectors (temporal binning for each sample).
        """
        if not isinstance(spike_activity, torch.Tensor):
            spike_activity = torch.tensor(spike_activity)

        # Determine the length of spike_activity and the necessary padding
        num_time_points = spike_activity.size(1)
        remainder = num_time_points % self.bin_size

        if remainder != 0:
            padding = self.bin_size - remainder
            spike_activity = torch.nn.functional.pad(spike_activity, (0, padding))

        binned_data = spike_activity.unfold(1, self.bin_size, self.bin_size).sum(dim=2)
        flat_binned = binned_data.view(binned_data.size(0), -1)        
        return flat_binned

class TemporalBinning(Sampler):
    """
    Sampler that calculates the temporal binning from spike activity.
    """

    def __init__(self, bin_size=10):
        """
        Initializes the TemporalBinning sampler.

        Parameters:
            bin_size (int): Size of the bin (default: 10).
        """
        self.bin_size = bin_size

    def sample(self, spike_activity):
        """
        Calculates the temporal binning from spike activity.

        Parameters:
            spike_activity (torch.Tensor): Spike activity tensor or array-like object.

        Returns:
            torch.Tensor: State vectors (temporal binning for each sample).
        """
        if not isinstance(spike_activity, torch.Tensor):
            spike_activity = torch.tensor(spike_activity)

        num_time_points = spike_activity.size(1)
        remainder = num_time_points % self.bin_size

        if remainder != 0:
            padding = self.bin_size - remainder
            spike_activity = torch.nn.functional.pad(spike_activity, (0, 0, 0, padding))  # Correct padding

        reshaped = spike_activity.reshape(spike_activity.size(0), -1, self.bin_size, spike_activity.size(2))
        binned_data = reshaped.sum(dim=2)

        flat_binned = binned_data.view(binned_data.size(0), -1)
        return flat_binned


class ISIstats(Sampler):
    """
    Sampler that calculates the interspike interval statistics from spike activity.
    """

    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        """
        Calculates the interspike interval statistics from spike activity.

        Parameters:
            spike_activity: Spike activity tensor or array-like object.

        Returns:
            State vectors (interspike interval statistics for each sample).
        """
        time_steps = torch.arange(spike_activity.shape[1]).unsqueeze(0).unsqueeze(2).expand(spike_activity.shape[0], -1, spike_activity.shape[2])
        spike_indices = time_steps * spike_activity
        spike_indices[spike_indices == 0] = float('inf')
        sorted_spike_indices, _ = torch.sort(spike_indices, dim=1)
        isi = torch.diff(sorted_spike_indices, dim=1)
        isi[isi == float('inf')] = float('nan')
        isi_mean = torch.nanmean(isi, dim=1)
        isi_mean[torch.isnan(isi_mean)] = -1
        return isi_mean
    
class DeSNN(Sampler):
    """
    DeSNN Sampler to sample state vectors based on spike activity.
    """

    def __init__(self, alpha=5, mod=0.8, drift_up=0.8, drift_down=0.01):
        """
        Initializes the DeSNN sampler.

        Parameters:
            alpha: Maximum initial weight (default: 5).
            mod: Modulation factor for importance of the order of spikes (default: 0.8).
            drift_up: Drift factor for spikes (default: 0.8).
            drift_down: Drift factor for no spikes (default: 0.01).
        """
        self.alpha = alpha
        self.mod = mod
        self.drift_up = drift_up
        self.drift_down = drift_down

    def sample(self, spike_activity):
        """
        Calculates ranks based on spike activity.

        Parameters:
            spike_activity: Spike activity tensor or array-like object.

        Returns:
            State vectors (DeSNN).
        """
        # Find the index of the first non-zero element along the second axis
        first_spike = (spike_activity != 0).int().argmax(axis=1)

        # Calculate initial ranks based on the alpha and mod parameters
        initial_ranks = self.alpha * (self.mod ** first_spike)

        # Calculate drift_up by multiplying spike activity along the second axis
        up = self.drift_up * spike_activity.sum(axis=1)

        # Calculate drift_down by multiplying by no spikes
        down = self.drift_down * (spike_activity.shape[1] - spike_activity.sum(axis=1))

        # Return the result of the rank calculation
        return initial_ranks + up - down
