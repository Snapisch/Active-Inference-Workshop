import math
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two normal distributions.
    
    KL(P||Q) = D_KL(N(μ₁,σ₁²)||N(μ₂,σ₂²))
             = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    
    Parameters:
    mu1 (float): Mean of the first (reference) distribution P
    sigma1 (float): Standard deviation of the first distribution P
    mu2 (float): Mean of the second distribution Q
    sigma2 (float): Standard deviation of the second distribution Q
    
    Returns:
    float: The KL divergence D_KL(P||Q)
    """
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("Standard deviations must be positive")

    return math.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def plot_kl_divergence_examples():
    # Example 1: Fixed second distribution, vary first distribution's mean
    mu2, sigma2 = 0.0, 1.0  # Reference distribution N(0,1)
    sigma1 = 1.0  # Keep same variance
    mu1_values = np.linspace(-3, 3, 100)
    
    kl_values_mean = [kl_divergence_normal(mu1, sigma1, mu2, sigma2) for mu1 in mu1_values]
    
    # Example 2: Fixed second distribution, vary first distribution's std
    mu1 = 0.0  # Keep same mean
    sigma1_values = np.linspace(0.1, 3, 100)
    
    kl_values_std = [kl_divergence_normal(mu1, sigma1, mu2, sigma2) for sigma1 in sigma1_values]
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: KL divergence vs mean difference
    plt.subplot(1, 2, 1)
    plt.plot(mu1_values, kl_values_mean, 'b-', linewidth=2)
    plt.xlabel('Mean of P (μ₁)')
    plt.ylabel('KL(P||Q)')
    plt.title('KL Divergence vs Mean\n(P=N(μ₁,1), Q=N(0,1))')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: KL divergence vs standard deviation ratio
    plt.subplot(1, 2, 2)
    plt.plot(sigma1_values, kl_values_std, 'r-', linewidth=2)
    plt.xlabel('Standard Deviation of P (σ₁)')
    plt.ylabel('KL(P||Q)')
    plt.title('KL Divergence vs Std Dev\n(P=N(0,σ₁), Q=N(0,1))')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_kl_heatmap():
    """
    Create a heatmap showing KL divergence for different parameter combinations.
    """
    # Fixed reference distribution Q = N(0, 1)
    mu2, sigma2 = 0.0, 1.0
    
    # Vary parameters of P
    mu1_range = np.linspace(-2, 2, 50)
    sigma1_range = np.linspace(0.2, 3, 50)
    
    mu1_grid, sigma1_grid = np.meshgrid(mu1_range, sigma1_range)
    
    # Calculate KL divergences
    kl_grid = np.zeros_like(mu1_grid)
    for i in range(len(sigma1_range)):
        for j in range(len(mu1_range)):
            kl_grid[i, j] = kl_divergence_normal(mu1_grid[i, j], sigma1_grid[i, j], mu2, sigma2)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(kl_grid, extent=[mu1_range.min(), mu1_range.max(), 
                                    sigma1_range.min(), sigma1_range.max()], 
                    aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='KL(P||Q)')
    plt.xlabel('Mean of P (μ₁)')
    plt.ylabel('Standard Deviation of P (σ₁)')
    plt.title('KL Divergence Heatmap\nP = N(μ₁,σ₁²), Q = N(0,1)')
    
    # Add contour lines
    contours = plt.contour(mu1_grid, sigma1_grid, kl_grid, colors='white', alpha=0.6, linewidths=0.5)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # Mark the reference point where P = Q
    plt.plot(0, 1, 'r*', markersize=15, label='P = Q (KL = 0)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kl_divergence_examples()
    create_kl_heatmap()