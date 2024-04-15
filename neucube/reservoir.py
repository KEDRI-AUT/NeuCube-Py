import torch
from tqdm import tqdm
import math
from .topology import small_world_connectivity
from .utils import print_summary
from .training import STDP

class Reservoir():
  def __init__(self, cube_shape=(10,10,10), inputs=None, coordinates=None, mapping=None, c=0.4, l=0.169, c_in=0.9, l_in=1.2, use_mps=False):
    """
    Initializes the reservoir object.

    Parameters:
        cube_shape (tuple): Dimensions of the reservoir as a 3D cube (default: (10,10,10)).
        inputs (int): Number of input features.
        coordinates (torch.Tensor): Coordinates of the neurons in the reservoir.
                                    If not provided, the coordinates are generated based on `cube_shape`.
        mapping (torch.Tensor): Coordinates of the input neurons.
                                If not provided, random connectivity is used.
        c (float): Parameter controlling the connectivity of the reservoir.
        l (float): Parameter controlling the connectivity of the reservoir.
        c_in (float): Parameter controlling the connectivity of the input neurons.
        l_in (float): Parameter controlling the connectivity of the input neurons.
        use_mps (bool): use Metal Performance Shaders (MPS) for Apple Silicon (if available).
    """
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    if torch.backends.mps.is_available() and use_mps is True:
      self.device = torch.device("mps:0")

    if coordinates is None:
      self.n_neurons = math.prod(cube_shape)
      x, y, z = torch.meshgrid(torch.linspace(0, 1, cube_shape[0]), torch.linspace(0, 1, cube_shape[1]), torch.linspace(0, 1, cube_shape[2]), indexing='xy')
      self.pos = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(self.device)
    else:
      self.n_neurons = coordinates.shape[0]
      self.pos = coordinates

    dist = torch.cdist(self.pos, self.pos)
    conn_mat = small_world_connectivity(dist, c=c, l=l) / 100
    inh_n = torch.randint(self.n_neurons, size=(int(self.n_neurons*0.2),))
    conn_mat[:, inh_n] = -conn_mat[:, inh_n]

    if mapping is None:
      input_conn = torch.where(torch.rand(self.n_neurons, inputs) > 0.95, torch.ones_like(torch.rand(self.n_neurons, inputs)), torch.zeros(self.n_neurons, inputs)) / 50
    else:
      dist_in = torch.cdist(coordinates, mapping, p=2)
      input_conn = small_world_connectivity(dist_in, c=c_in, l=l_in) / 50

    self.w_latent = conn_mat.to(self.device)
    self.w_in = input_conn.to(self.device)

  def simulate(self, X, mem_thr=0.1, refractory_period=5, train=True, learning_rule=STDP(), verbose=True):
    """
    Simulates the reservoir activity given input data.

    Parameters:
        X (torch.Tensor): Input data of shape (batch_size, n_time, n_features).
        mem_thr (float): Membrane threshold for spike generation.
        refractory_period (int): Refractory period after a spike.
        train (bool): Flag indicating whether to perform online training of the reservoir.
        learning_rule (LearningRule): The learning rule implementation to use for training.
        verbose (bool): Flag indicating whether to display progress during simulation.

    Returns:
        torch.Tensor: Spike activity of the reservoir neurons over time, of shape (batch_size, n_time, n_neurons).

    Raises:
        Exception: If learning rule implementation is not specified and training is enabled
    """
    if train is True and learning_rule is None:
      raise Exception("learning rule implementation must be specified if training is enabled")

    self.batch_size, self.n_time, self.n_features = X.shape

    spike_rec = torch.zeros(self.batch_size, self.n_time, self.n_neurons)

    if train is True:
      learning_rule.setup(self.device, self.n_neurons)

    for s in tqdm(range(X.shape[0]), disable = not verbose):

      spike_latent = torch.zeros(self.n_neurons).to(self.device)
      mem_poten = torch.zeros(self.n_neurons).to(self.device)
      refrac = torch.ones(self.n_neurons).to(self.device)
      refrac_count = torch.zeros(self.n_neurons).to(self.device)
      spike_times = torch.zeros(self.n_neurons).to(self.device)

      if train is True:
        learning_rule.per_sample(s)

      for k in range(self.n_time):
        spike_in = X[s,k,:]
        spike_in = spike_in.to(self.device)

        refrac[refrac_count < 1] = 1

        I = torch.sum(self.w_in*spike_in, axis=1)+torch.sum(self.w_latent*spike_latent, axis=1)
        mem_poten = mem_poten*torch.exp(torch.tensor(-(1/40)))*(1-spike_latent)+(refrac*I)

        spike_latent[mem_poten >= mem_thr] = 1
        spike_latent[mem_poten < mem_thr] = 0

        refrac[mem_poten >= mem_thr] = 0
        refrac_count[mem_poten >= mem_thr] = refractory_period
        refrac_count = refrac_count-1

        if train is True:
          learning_rule.per_time_slice(s, k)
          pre_updates, pos_updates = learning_rule.train(k-spike_times, self.w_latent, spike_latent)
          self.w_latent += pre_updates
          self.w_latent += pos_updates
          learning_rule.reset()

        spike_times[mem_poten >= mem_thr] = k
        
        spike_rec[s,k,:] = spike_latent

    return spike_rec

  def summary(self):
    """
    Prints a summary of the reservoir.
    """
    res_info = [["Neurons", str(self.n_neurons)],
                ["Reservoir connections", str(sum(sum(self.w_latent != 0)).item())],
                ["Input connections", str(sum(sum(self.w_in != 0)).item())],
                ["Device", str(self.device)]]

    print_summary(res_info)