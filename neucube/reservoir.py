import torch
from tqdm import tqdm
import math
from numpy.random import uniform as unif
from .topology import small_world_connectivity
from .utils import print_summary

# NRDP constants
# fine-tuning is necessary for better classification results
MIN_A = 0.0
MAX_A = 0.901679611594126
GAIN_A = 0.5428717518563672
MIN_N = 0.0
MAX_N = 0.23001290732040292
GAIN_N = 0.011660312977761912
MIN_GA = 0.0
MAX_GA = 0.7554145024515596
GAIN_GA = 0.3859076787035615
MIN_GB = 0.0
MAX_GB = 0.7954714253083993
GAIN_GB = 0.11032115434326673

def is_gaba_activated(gaba_rate=0.7):
  return unif(0, 1) < gaba_rate

def get_gaba_gain():
  return GAIN_GA if is_gaba_activated() else GAIN_GB

def get_gaba_min():
  return MIN_GA if is_gaba_activated() else MIN_GB

def get_gaba_max():
  return MAX_GA if is_gaba_activated() else MAX_GB

class Reservoir():
  def __init__(self, cube_shape=(10,10,10), inputs=None, coordinates=None, mapping=None, c=1.2, l=1.6, c_in = 0.9, l_in = 1.2):
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
    """
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    # use Metal Performance Shaders (beta)
    #self.device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu:0")

    if coordinates is None:
      self.n_neurons = math.prod(cube_shape)
      x, y, z = torch.meshgrid(torch.linspace(0, 1, cube_shape[0]), torch.linspace(0, 1, cube_shape[1]), torch.linspace(0, 1, cube_shape[2]), indexing='xy')
      pos = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
    else:
      self.n_neurons = coordinates.shape[0]
      pos = coordinates

    dist = torch.cdist(pos, pos)
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

  def simulate(self, X, mem_thr=0.1, refractory_period=5, train=True, verbose=True):
    """
    Simulates the reservoir activity given input data.

    Parameters:
        X (torch.Tensor): Input data of shape (batch_size, n_time, n_features).
        mem_thr (float): Membrane threshold for spike generation.
        refractory_period (int): Refractory period after a spike.
        train (bool): Flag indicating whether to perform online training of the reservoir.
        verbose (bool): Flag indicating whether to display progress during simulation.

    Returns:
        torch.Tensor: Spike activity of the reservoir neurons over time, of shape (batch_size, n_time, n_neurons).
    """
    self.batch_size, self.n_time, self.n_features = X.shape

    spike_rec = torch.zeros(self.batch_size, self.n_time, self.n_neurons)

    # iterate over each sample
    for s in tqdm(range(X.shape[0]), disable = not verbose):

      spike_latent = torch.zeros(self.n_neurons).to(self.device)
      mem_poten = torch.zeros(self.n_neurons).to(self.device)
      refrac = torch.ones(self.n_neurons).to(self.device)
      refrac_count = torch.zeros(self.n_neurons).to(self.device)
      spike_times = torch.zeros(self.n_neurons).to(self.device)

      # NRDP specific
      firing_state = torch.zeros(self.n_neurons).to(self.device)
      a_state = torch.zeros(self.n_neurons).to(self.device)
      a_t = torch.zeros(self.n_neurons).to(self.device)
      n_t = torch.zeros(self.n_neurons).to(self.device)
      g_t = torch.zeros(self.n_neurons).to(self.device)
      time_window = 10
      gaba_impact = 0.01

      # iterate over each row of data (time steps)
      for k in range(self.n_time):

        spike_in = X[s,k,:] # one row of data i.e. a time step
        spike_in = spike_in.to(self.device)

        refrac[refrac_count < 1] = 1

        # calculate membrane potential
        I = torch.sum(self.w_in*spike_in, axis=1)+torch.sum(self.w_latent*spike_latent, axis=1) # current
        mem_poten = mem_poten*torch.exp(torch.tensor(-(1/40)))*(1-spike_latent)+(refrac*I) # LIF

        # output spike
        spike_latent[mem_poten >= mem_thr] = 1 # spike
        spike_latent[mem_poten < mem_thr] = 0 # no spike

        refrac[mem_poten >= mem_thr] = 0 # reset after spike emitted
        refrac_count[mem_poten >= mem_thr] = refractory_period
        refrac_count = refrac_count-1

        if train is True:
          self.aux = k-spike_times

          #region STDP
          #t_constant = 3
          #self.pre_w = 0.0001*torch.exp(-self.aux/t_constant)*torch.gt(self.aux,0).int() # ltp
          #self.pos_w = -0.01*torch.exp(-self.aux/t_constant)*torch.gt(self.aux,0).int() # ltd
          #pre_updates = self.pre_w*torch.gt((self.w_latent.T*spike_latent).T, 0).int()
          #pos_updates = self.pos_w*torch.gt(self.w_latent*spike_latent, 0).int()
          #end region

          # region NRDP
          # NRDP implementation as per: https://ieeexplore.ieee.org/document/8118188
          a_t = torch.where(firing_state >= time_window,
                            torch.maximum(torch.full_like(a_t, MIN_A), a_t-(gaba_impact * get_gaba_gain())),
                            torch.minimum(torch.full_like(a_t, MAX_A), a_t+GAIN_A))
          firing_state[spike_latent < 1] += 1
          # TODO: set A(t) to MAX_A for duration of time window
          a_t = torch.where(a_t >= MAX_A,
                            torch.where(a_state < time_window, MAX_A, a_t), a_t)
          a_state[a_t >= MAX_A] += 1
          # only calculate N(t) when A(t) reaches its max level and there is a spike
          n_t = torch.where(a_t >= MAX_A,
                            torch.where(spike_latent > 0,
                                        torch.maximum(torch.full_like(n_t, MIN_N), n_t-GAIN_N),
                                        torch.minimum(torch.full_like(n_t, MAX_N), n_t+GAIN_N)),
                            MIN_N)
          g_t = torch.where(spike_latent > 0,
                            torch.maximum(torch.full_like(g_t, get_gaba_min()), g_t-get_gaba_gain()),
                            torch.minimum(torch.full_like(g_t, get_gaba_max()), g_t+get_gaba_gain()*self.aux))

          # reset
          firing_state[firing_state >= time_window] = 0
          a_state[a_state >= time_window] = 0

          pre_updates = torch.zeros(self.n_neurons, self.n_neurons)
          pos_updates = (n_t + a_t) - g_t # Wij
          # endregion

          self.w_latent += pre_updates
          self.w_latent += pos_updates

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