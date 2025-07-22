import math
import numpy as np
import matplotlib.pyplot as plt

def calc_differential_entropy_norm(u, o):
    """
    Calculate the differential entropy of a normal distribution.

    Parameters:
    u (float): Mean of the normal distribution.
    o (float): Standard deviation of the normal distribution.

    Returns:
    float: The differential entropy of the normal distribution.
    """

    return 0.5 * math.log(2 * math.pi * math.exp(1) * o**2) # analytical solution

if __name__ == "__main__":
    means = np.arange(-5, 6, 1)  # Mean values from -5 to 5
    std_devs = np.arange(0.1, 4.1, 0.25)  # Standard deviations from 0.1 to 4.0

    std_dev = 1.0  # Fixed standard deviation
    entropies = []
    for mean in means:
        entropy = calc_differential_entropy_norm(mean, std_dev)
        entropies.append(entropy)

    mean = 1.0  # Fixed mean
    entropies_std = []
    for std_dev in std_devs:
        entropy = calc_differential_entropy_norm(mean, std_dev)
        entropies_std.append(entropy)

    plt.figure(figsize=(15, 6))
    
    # Left plot: Entropy vs Mean
    plt.subplot(1, 2, 1)
    plt.plot(means, entropies, marker='o', color='blue', linewidth=2, markersize=6)
    plt.title('Differential Entropy vs Mean\n(σ = 1.0)')
    plt.xlabel('Mean (μ)')
    plt.ylabel('Differential Entropy')
    plt.grid(True, alpha=0.3)
    
    # Right plot: Entropy vs Standard Deviation
    plt.subplot(1, 2, 2)
    plt.plot(std_devs, entropies_std, marker='o', color='red', linewidth=2, markersize=6)
    plt.title('Differential Entropy vs Standard Deviation\n(μ = 1.0)')
    plt.xlabel('Standard Deviation (σ)')
    plt.ylabel('Differential Entropy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()