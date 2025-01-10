import torch
from abc import ABC, abstractmethod

class Encoder(ABC):
  def __init__(self):
    super().__init__()

  def encode_dataset(self, dataset):
    """
    Encodes a dataset using the implemented encoding technique.

    Args:
      dataset (torch.Tensor): Input dataset to be encoded.

    Returns:
      torch.Tensor: Encoded dataset with spike patterns.
    """
    encoded_data = torch.zeros_like(dataset)

    for i in range(dataset.shape[0]):
      for j in range(dataset.shape[2]):
        encoded_data[i][:, j] = self.encode(dataset[i][:, j])

    return encoded_data

  @abstractmethod
  def encode(self, X_in):
    """
    Encodes an input sample using the implemented encoding technique.

    Args:
      X_in (torch.Tensor): Input sample to be encoded.

    Returns:
      torch.Tensor: Encoded spike pattern for the input sample.
    """
    pass

class Delta(Encoder):
  def __init__(self, threshold=0.1):
    """
    Initializes the Delta encoder with a threshold value.

    Args:
      threshold (float, optional): Threshold value for spike generation. Defaults to 0.1.
    """
    self.threshold = threshold

  def encode(self, sample):
    """
    Encodes an input sample using delta encoding.

    Delta encoding compares each element in the sample with its previous element,
    and if the difference exceeds the threshold, it generates a spike (1); otherwise, no spike (0).

    Args:
      sample (torch.Tensor): Input sample to be encoded.

    Returns:
      torch.Tensor: Encoded spike train for the input sample.
    """
    aux = torch.cat((sample[0].unsqueeze(0), sample))[:-1]
    spikes = torch.ones_like(sample) * (sample - aux >= self.threshold)
    return spikes

class StepForward(Encoder):
  def __init__(self, threshold=0.1):
    """
    Initializes the StepForward encoder with a threshold value.

    Args:
      threshold (float, optional): Threshold value for spike generation. Defaults to 0.1.
    """
    super().__init__()
    self.threshold = threshold

  def encode(self, sample):
    spikes = torch.zeros_like(sample, dtype=torch.int8)
    base = sample[0]
    for t in range(1, sample.size(0)):
      if sample[t] >= base + self.threshold:
        spikes[t] = 1
        base += self.threshold
      elif sample[t] <= base - self.threshold:
        spikes[t] = -1
        base -= self.threshold
    return spikes
