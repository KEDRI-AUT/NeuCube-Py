import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import neucube

def spike_raster(spike_activity, fig_size=(10, 5)):
    """
    Generate a raster plot of spike activity.

    Parameters:
    spike_activity (torch.Tensor): 2D tensor representing spike activity, where each column corresponds to a neuron and each row corresponds to a time step.
    fig_size (tuple, optional): A tuple specifying the figure size. Default is (10, 5).

    Raises:
    ValueError: If the input is not a 2D tensor.

    Returns:
    None
    """

    # Check if input is a 2D tensor
    if not isinstance(spike_activity, torch.Tensor) or len(spike_activity.shape) != 2:
        raise ValueError('Input should be a 2D tensor')
    
    # Find indices of non-zero elements (spikes)
    x, y = torch.where(spike_activity)
    
    # Create scatter plot
    plt.figure(figsize=fig_size)
    plt.scatter(x, y, marker='o', s=1)
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron Index')
    plt.show()

def plot_connections(reservoir, exci_thres=0.002, inhi_thres=-0.003, fig_size=(10, 8)):
    """
    Plot the connections within a NeuCube reservoir.

    Parameters:
    - reservoir (neucube.Reservoir): NeuCube Reservoir object.
    - exci_thres (float): Excitatory threshold for plotting connections. Default is 0.002.
    - inhi_thres (float): Inhibitory threshold for plotting connections. Default is -0.003.
    - fig_size (tuple): Figure size (width, height) in inches. Default is (10, 8).

    Raises:
    - ValueError: If the input is not a NeuCube Reservoir object.
    """

    # Check if the input is a NeuCube Reservoir object
    if not isinstance(reservoir, neucube.Reservoir):
        raise ValueError('Input should be a NeuCube Reservoir object')

    exci_vis_conn = torch.where(reservoir.w_latent > exci_thres)
    inh_vis_conn = torch.where((reservoir.w_latent != 0) & (reservoir.w_latent < inhi_thres))

    # Get positions of parent and child neurons for excitatory connections
    exci_parent, exci_children = reservoir.pos[exci_vis_conn[0]], reservoir.pos[exci_vis_conn[1]]
    # Get positions of parent and child neurons for inhibitory connections
    inh_parent, inh_children = reservoir.pos[inh_vis_conn[0]], reservoir.pos[inh_vis_conn[1]]

    exci_x, exci_b, exci_c = [torch.hstack((exci_parent[:,[i]], exci_children[:,[i]])).cpu() for i in range(3)] 
    inhi_a, inhi_b, inhi_c = [torch.hstack((inh_parent[:,[i]], inh_children[:,[i]])).cpu() for i in range(3)] 

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*reservoir.pos.T.cpu(), c='black', marker='o')

    exci_lines = np.stack([exci_x, exci_b, exci_c], axis=-1)
    ax.add_collection3d(Line3DCollection(exci_lines, colors='blue'))
    inhi_lines = np.stack([inhi_a, inhi_b, inhi_c], axis=-1)
    ax.add_collection3d(Line3DCollection(inhi_lines, colors='red'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()