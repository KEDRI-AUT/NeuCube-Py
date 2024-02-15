import torch
from tqdm import tqdm
import math
from .topology import small_world_connectivity
from .utils import print_summary
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    self.coordinates = coordinates
    self.mapping = mapping


  def retrieve_conn_mat(self):
    """
    Retrieves the connection matrix established after small world connectivity, and converts 
    it into a csv file to be saved.
    """
    mat = self.w_latent
    DF = pd.DataFrame(mat.cpu())
    DF.to_csv("conn.csv")



  def visualize_cube(self,cube_shape=(10,10,10),coordinates=None, mapping=None):
    """
      Visualises the cube in a 3D space, indicating the input neurons and their positions

      Parameters:
        cube_shape(tuple): Dimensions of the cube
        coordinates(torch.Tensor): Coordinates of the neurons in the reservoir.
                                    If not provided, the coordinates were generated based on `cube_shape`.
        mapping (torch.Tensor): Coordinates of the input neurons.
                                If not provided, random connectivity was used.        
    """
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(111, projection='3d')
    if coordinates is None:
      x, y, z = torch.meshgrid(torch.linspace(0, 1, cube_shape[0]), torch.linspace(0, 1, cube_shape[1]), torch.linspace(0, 1, cube_shape[2]), indexing='xy')
      ax.scatter(x.flatten(), y.flatten(), z.flatten(), s = 8, c='#3258a8')  #f5d1b6 #957d5f #A18A6C

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.grid(False)
      plt.show()
    else:
      coordinates_np = coordinates.numpy()
      ax.scatter(coordinates_np[:,1], coordinates_np[:,0], coordinates_np[:,2], s = 8, c='#3258a8', zorder = 0)
      if mapping is not None:
        mapping_np = mapping.numpy()
        ax.scatter(mapping_np[:,1], mapping_np[:,0], mapping_np[:,2], s =60, c = 'black', zorder = 10 )


      ax.set_xlabel('Y')
      ax.set_ylabel('X')
      ax.set_zlabel('Z')
      ax.invert_xaxis()
      ax.grid(False)
      plt.show()

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

    for s in tqdm(range(X.shape[0]), disable = not verbose):  #range is from 0 to 59, samples

      spike_latent = torch.zeros(self.n_neurons).to(self.device)
      mem_poten = torch.zeros(self.n_neurons).to(self.device)
      refrac = torch.ones(self.n_neurons).to(self.device)
      refrac_count = torch.zeros(self.n_neurons).to(self.device)
      spike_times = torch.zeros(self.n_neurons).to(self.device)

      for k in range(self.n_time):    # k goes from 0 to 127 (timestamps)

        spike_in = X[s,k,:]     #spike input for all 14 features
        spike_in = spike_in.to(self.device)

        refrac[refrac_count < 1] = 1

        I = torch.sum(self.w_in*spike_in, axis=1)+torch.sum(self.w_latent*spike_latent, axis=1)
        mem_poten = mem_poten*torch.exp(torch.tensor(-(1/40)))*(1-spike_latent)+(refrac*I)

        spike_latent[mem_poten >= mem_thr] = 1
        spike_latent[mem_poten < mem_thr] = 0

        refrac[mem_poten >= mem_thr] = 0
        refrac_count[mem_poten >= mem_thr] = refractory_period
        refrac_count = refrac_count-1

        if train == True:
          t_constant = 3
          self.aux = k-spike_times
          self.pre_w = 0.0001*torch.exp(-self.aux/t_constant)*torch.gt(self.aux,0).int()
          self.pos_w = -0.01*torch.exp(-self.aux/t_constant)*torch.gt(self.aux,0).int()
          pre_updates = self.pre_w*torch.gt((self.w_latent.T*spike_latent).T, 0).int()
          pos_updates = self.pos_w*torch.gt(self.w_latent*spike_latent, 0).int()

          self.w_latent += pre_updates
          self.w_latent += pos_updates



        spike_times[mem_poten >= mem_thr] = k

        spike_rec[s,k,:] = spike_latent
        self.output  = spike_rec
    
    return spike_rec
  
  def post_weights(self):
    """
    Retrieves the weight matrix obtained after running the simulate function, and converts 
    it into a csv file to be saved.
    """
    mat = self.w_latent
    DF = pd.DataFrame(mat.cpu())
    DF.to_csv("post_conn.csv")

  def input_spike_count(self):
    """
    This caclulates the total spike count for input neurons over time, for each sample

    """
    mapping = self.mapping
    coordinates = self.coordinates
    out = self.output
    eeg_np = mapping.numpy()
    brain_np = coordinates.numpy()
    idx = []
    for row in eeg_np:
      mask = np.all(brain_np == row, axis=1)

      # Find the indices where the mask is True
      indices = np.where(mask)[0]

      idx.append(indices)

    indexi = [item[0] for item in idx]

    matrix = torch.zeros(out.shape[0], len(indexi))

    for sm in range (len(out)):
      for i in range (len(indexi)):
        matrix[sm][i] = out[sm][:,indexi[i]].sum().item()


    return matrix


  def feature_vectors(self, k_vec):
    """
    This function can be used to extract and store K feature vectors
     of length N that represent the number of spikes of each neurons
    from the cube within each of the k time intervals from T.

    """

    out = self.output
    window = int(out.shape[1]/k_vec)+1
    feature = torch.zeros(out.shape[0], int(k_vec), out.shape[2])

    for sm in range(out.shape[0]):

      for neuron in range(out.shape[2]):
        i = 0
        idx = 0
        while(i<out.shape[1]):
          if(i+window<out.shape[1]):
            feature[sm][idx][neuron] = out[sm][i:i+window-1,neuron].sum().numpy().item()
          else:
            feature[sm][idx][neuron] = out[sm][i:out.shape[1]-1,neuron].sum().numpy().item()
          idx+=1
          i = i+window

    return feature

  def plot_rasters(self, i):
    """
    Creates a raster plot for the given sample.

    Parameters:
        i(integer): Sample number (indexed from 0)
    """
    out = self.output
    spike_indices = np.transpose(out[i].nonzero())
    plt.figure(figsize=(10, 6))
    plt.scatter(spike_indices[0], spike_indices[1], s =0.1, color = 'black')
    plt.show()

  def summary(self):
    """
    Prints a summary of the reservoir.
    """
    res_info = [["Neurons", str(self.n_neurons)],
                ["Reservoir connections", str(sum(sum(self.w_latent != 0)).item())],
                ["Input connections", str(sum(sum(self.w_in != 0)).item())],
                ["Device", str(self.device)]]

    print_summary(res_info)