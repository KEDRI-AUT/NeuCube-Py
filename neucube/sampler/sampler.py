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