# Neucube Py

Neucube Py is a Python implementation of the Neucube architecture, a spiking neural network (SNN) model inspired by the brain's structural and functional principles. It is designed to capture and process patterns in spatio-temporal data.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

## Key Features

- Implementation of Neucube in Python
- Written with pytorch for faster computations
- Ability to capture and process patterns in spatio-temporal data
- Integration with other popular Python libraries

## Installation

To use Neucube Py, you can clone this repository:

```bash
git clone https://github.com/KEDRI-AUT/NeuCube-Py.git
```

## Getting Started

To start using Neucube Py, you can refer to the [examples](examples/) directory, which contains Jupyter notebooks demonstrating various use cases and applications of the Neucube algorithm.

You can also try out a running example using Google Colab by clicking the button below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UOHahwu4VDsh-Qq2R2-syObrAatCJygX?usp=sharing)

## Usage

The core functionality of Neucube Py revolves around the `reservoir` class, which represents the spiking neural network model. Here is a basic example of how to use Neucube Py:

```python
from neucube import Reservoir
from neucube.encoder import Delta
from neucube.sampler import SpikeCount

# Create a Reservoir 
res = Reservoir(inputs=14)

# Convert data to spikes
X = Delta().encode_dataset(data)

# Simulate the Reservior
out = res.simulate(X)

# Extract state vectors from the spiking activity
state_vec = SpikeCount.sample(out)

# Perform prediction and validation
# ...

```

## Contributing

Contributions to Neucube Py are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Before contributing, please read our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

Neucube Py is licensed under the [GNU AGPLv3](LICENSE.md).

## Acknowledgments

Neucube Py builds upon the original Neucube model developed at KEDRI Auckland University of Technology, New Zealand. We acknowledge their contributions to the field of spiking neural networks and their research that inspired this project.

## References

For more information about the Neucube algorithm and related research papers, please refer to the following publications:

- Original Neucube Paper: [NeuCube: A spiking neural network architecture for mapping, learning and understanding of spatio-temporal brain data](https://www.sciencedirect.com/science/article/abs/pii/S0893608014000070)
- Additional Research Papers: [List of Neucube-related publications](https://kedri.aut.ac.nz/our-projects/publications)

## Contact

For any inquiries or questions, please contact us at nkasabov@aut.ac.nz
