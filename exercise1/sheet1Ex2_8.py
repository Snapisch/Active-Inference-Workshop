import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize


class BimodalGaussian:
    """Class to represent a bimodal Gaussian distribution (mixture of two Gaussians)."""
    
    def __init__(self, mu1, sigma1, mu2, sigma2, weight=0.5):
        """
        Initialize bimodal Gaussian distribution.
        
        Parameters:
        mu1, sigma1: Parameters of first Gaussian
        mu2, sigma2: Parameters of second Gaussian  
        weight: Weight for first Gaussian (1-weight for second)
        """
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.weight = weight
        
    def pdf(self, x):
        """Calculate probability density function of bimodal distribution."""
        return (self.weight * norm.pdf(x, self.mu1, self.sigma1) + 
                (1 - self.weight) * norm.pdf(x, self.mu2, self.sigma2))
    
    def get_empirical_moments(self, x_range=None):
        """Calculate empirical mean and variance of the bimodal distribution."""
        if x_range is None:
            # Use a wide range based on the component distributions
            x_min = min(self.mu1 - 4*self.sigma1, self.mu2 - 4*self.sigma2)
            x_max = max(self.mu1 + 4*self.sigma1, self.mu2 + 4*self.sigma2)
            x_range = np.linspace(x_min, x_max, 10000)
        
        pdf_vals = self.pdf(x_range)
        dx = x_range[1] - x_range[0]
        
        # Normalize (should already be normalized, but just to be sure)
        pdf_vals = pdf_vals / (np.sum(pdf_vals) * dx)
        
        # Calculate moments
        mean = np.sum(x_range * pdf_vals * dx)
        variance = np.sum((x_range - mean)**2 * pdf_vals * dx)
        
        return mean, np.sqrt(variance)

def kl_divergence_continuous(bimodal_dist, mu_gauss, sigma_gauss, x_range=None):
    """
    Calculate KL divergence between bimodal distribution and single Gaussian.
    
    KL(P||Q) = ∫ P(x) log(P(x)/Q(x)) dx
    
    Parameters:
    bimodal_dist: BimodalGaussian object
    mu_gauss, sigma_gauss: Parameters of single Gaussian
    x_range: Range for numerical integration
    """
    if sigma_gauss <= 0:
        return np.inf
    
    if x_range is None:
        # Define integration range based on both distributions
        x_min = min(bimodal_dist.mu1 - 4*bimodal_dist.sigma1, 
                   bimodal_dist.mu2 - 4*bimodal_dist.sigma2,
                   mu_gauss - 4*sigma_gauss)
        x_max = max(bimodal_dist.mu1 + 4*bimodal_dist.sigma1, 
                   bimodal_dist.mu2 + 4*bimodal_dist.sigma2,
                   mu_gauss + 4*sigma_gauss)
        x_range = np.linspace(x_min, x_max, 10000)
    
    # Calculate PDFs
    p_x = bimodal_dist.pdf(x_range)
    q_x = norm.pdf(x_range, mu_gauss, sigma_gauss)
    
    # Avoid log(0) by filtering out zero probabilities
    mask = (p_x > 1e-15) & (q_x > 1e-15)
    
    if np.sum(mask) == 0:
        return np.inf
    
    # Calculate KL divergence using numerical integration
    integrand = p_x[mask] * np.log(p_x[mask] / q_x[mask])
    dx = x_range[1] - x_range[0]
    kl_div = np.sum(integrand) * dx
    
    return kl_div

def objective_function(params, bimodal_dist):
    """Objective function for optimization - KL divergence to minimize."""
    mu, sigma = params
    return kl_divergence_continuous(bimodal_dist, mu, sigma)

def find_optimal_gaussian(bimodal_dist, method='L-BFGS-B'):
    """
    Find optimal Gaussian parameters that minimize KL divergence.
    
    Parameters:
    bimodal_dist: BimodalGaussian distribution
    method: optimization method
    
    Returns:
    result: optimization result object
    """
    # Initial guess based on empirical moments
    # So first and 2nd moment of the gaussian will match the bimodal distribution,
    # which already reduces KL divergence greatly
    empirical_mean, empirical_std = bimodal_dist.get_empirical_moments()
    initial_guess = [empirical_mean, empirical_std]
    
    print(f"Initial guess - Mean: {empirical_mean:.3f}, Std: {empirical_std:.3f}")
    
    # Bounds: mean can be anywhere, sigma must be positive
    bounds = [(None, None), (0.000001, None)]
    
    # Minimize KL divergence
    result = minimize(
        objective_function,
        initial_guess,
        args=(bimodal_dist,),
        method=method,
        bounds=bounds
    )
    
    return result

def create_comprehensive_analysis():
    """Create comprehensive analysis of bimodal-normal KL divergence."""
    
    # Create bimodal Gaussian distribution
    # Two well-separated peaks
    bimodal = BimodalGaussian(mu1=-2, sigma1=0.8, mu2=3, sigma2=1.2, weight=0.6)
    
    print("Bimodal Gaussian Distribution Analysis")
    print("=" * 50)
    print(f"Component 1: N({bimodal.mu1}, {bimodal.sigma1}²) with weight {bimodal.weight}")
    print(f"Component 2: N({bimodal.mu2}, {bimodal.sigma2}²) with weight {1-bimodal.weight}")
    
    # Calculate empirical statistics
    empirical_mean, empirical_std = bimodal.get_empirical_moments()
    print(f"\nEmpirical moments:")
    print(f"Mean: {empirical_mean:.3f}")
    print(f"Std:  {empirical_std:.3f}")
    
    # Find optimal Gaussian parameters
    print("\nOptimizing Gaussian parameters...")
    result = find_optimal_gaussian(bimodal)
    
    if result.success:
        optimal_mu, optimal_sigma = result.x
        optimal_kl = result.fun
        
        print(f"\nOptimization successful!")
        print(f"Optimal μ: {optimal_mu:.3f}")
        print(f"Optimal σ: {optimal_sigma:.3f}")
        print(f"Minimal KL divergence: {optimal_kl:.6f}")
    else:
        print("Optimization failed!")
        optimal_mu, optimal_sigma = empirical_mean, empirical_std
        optimal_kl = kl_divergence_continuous(bimodal, optimal_mu, optimal_sigma)
    
    plt.figure(figsize=(16, 12))
    
    # Define x range for plotting
    x_plot = np.linspace(bimodal.mu1 - 4*bimodal.sigma1, 
                        bimodal.mu2 + 4*bimodal.sigma2, 1000)
    
    # Plot 1: Distribution comparison
    plt.subplot(2, 3, 1)
    bimodal_pdf = bimodal.pdf(x_plot)
    optimal_pdf = norm.pdf(x_plot, optimal_mu, optimal_sigma)
    empirical_pdf = norm.pdf(x_plot, empirical_mean, empirical_std)
    
    plt.plot(x_plot, bimodal_pdf, 'b-', linewidth=2, label='Bimodal Gaussian')
    plt.plot(x_plot, optimal_pdf, 'r-', linewidth=2, 
             label=f'Optimal N({optimal_mu:.1f},{optimal_sigma:.1f})')
    plt.plot(x_plot, empirical_pdf, 'g--', linewidth=2, 
             label=f'Empirical N({empirical_mean:.1f},{empirical_std:.1f})')
    
    # Mark component means
    plt.axvline(bimodal.mu1, color='blue', linestyle=':', alpha=0.7, label='Component means')
    plt.axvline(bimodal.mu2, color='blue', linestyle=':', alpha=0.7)
    
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: KL divergence vs mean (fixed optimal sigma)
    plt.subplot(2, 3, 2)
    mean_range = np.linspace(bimodal.mu1 - 2, bimodal.mu2 + 2, 50)
    kl_values_mean = [kl_divergence_continuous(bimodal, mu, optimal_sigma) for mu in mean_range]
    plt.plot(mean_range, kl_values_mean, 'b-', linewidth=2)
    plt.plot(optimal_mu, optimal_kl, 'ro', markersize=10)
    plt.axvline(empirical_mean, color='green', linestyle='--', alpha=0.7, label='Empirical mean')
    plt.xlabel('Gaussian Mean (μ)')
    plt.ylabel('KL Divergence')
    plt.title(f'KL vs Mean (σ={optimal_sigma:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: KL divergence vs sigma (fixed optimal mean)
    plt.subplot(2, 3, 3)
    sigma_range = np.linspace(0.5, 4, 50)
    kl_values_sigma = [kl_divergence_continuous(bimodal, optimal_mu, sigma) for sigma in sigma_range]
    plt.plot(sigma_range, kl_values_sigma, 'r-', linewidth=2)
    plt.plot(optimal_sigma, optimal_kl, 'ro', markersize=10)
    plt.axvline(empirical_std, color='green', linestyle='--', alpha=0.7, label='Empirical std')
    plt.xlabel('Gaussian Std (σ)')
    plt.ylabel('KL Divergence')
    plt.title(f'KL vs Std (μ={optimal_mu:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: 2D heatmap of KL divergence
    plt.subplot(2, 3, 4)
    mu_range_2d = np.linspace(bimodal.mu1 - 1, bimodal.mu2 + 1, 25)
    sigma_range_2d = np.linspace(0.8, 3.5, 25)
    mu_grid, sigma_grid = np.meshgrid(mu_range_2d, sigma_range_2d)
    
    print("Computing 2D KL divergence heatmap...")
    kl_grid = np.zeros_like(mu_grid)
    for i in range(len(sigma_range_2d)):
        for j in range(len(mu_range_2d)):
            kl_grid[i, j] = kl_divergence_continuous(bimodal, mu_grid[i, j], sigma_grid[i, j])
    
    im = plt.imshow(kl_grid, extent=[mu_range_2d.min(), mu_range_2d.max(), 
                                    sigma_range_2d.min(), sigma_range_2d.max()], 
                    aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='KL Divergence')
    plt.plot(optimal_mu, optimal_sigma, 'r*', markersize=15, label='Optimal')
    plt.plot(empirical_mean, empirical_std, 'g^', markersize=12, label='Empirical')
    plt.xlabel('Mean (μ)')
    plt.ylabel('Std (σ)')
    plt.title('KL Divergence Heatmap')
    plt.legend()
    
    # Plot 5: Component distributions
    plt.subplot(2, 3, 5)
    comp1_pdf = bimodal.weight * norm.pdf(x_plot, bimodal.mu1, bimodal.sigma1)
    comp2_pdf = (1-bimodal.weight) * norm.pdf(x_plot, bimodal.mu2, bimodal.sigma2)
    plt.plot(x_plot, comp1_pdf, 'c-', linewidth=2, 
             label=f'Component 1 (w={bimodal.weight})')
    plt.plot(x_plot, comp2_pdf, 'm-', linewidth=2, 
             label=f'Component 2 (w={1-bimodal.weight})')
    plt.plot(x_plot, bimodal_pdf, 'b-', linewidth=2, label='Sum (Bimodal)')
    plt.xlabel('x')
    plt.ylabel('Weighted Density')
    plt.title('Bimodal Components')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()
    
    return {
        'bimodal': bimodal,
        'optimal_mu': optimal_mu,
        'optimal_sigma': optimal_sigma,
        'optimal_kl': optimal_kl,
        'empirical_mean': empirical_mean,
        'empirical_std': empirical_std
    }

if __name__ == "__main__":
    # Run comprehensive analysis
    results = create_comprehensive_analysis()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Bimodal distribution components:")
    print(f"  Component 1: N({results['bimodal'].mu1}, {results['bimodal'].sigma1}²)")
    print(f"  Component 2: N({results['bimodal'].mu2}, {results['bimodal'].sigma2}²)")
    print(f"  Mixing weight: {results['bimodal'].weight:.1f} / {1-results['bimodal'].weight:.1f}")
    
    print(f"\nBest approximating Gaussian:")
    print(f"  Mean (μ): {results['optimal_mu']:.3f}")
    print(f"  Std (σ):  {results['optimal_sigma']:.3f}")
    print(f"  KL divergence: {results['optimal_kl']:.6f}")
    
    print(f"\nComparison with empirical moments:")
    print(f"  Empirical mean: {results['empirical_mean']:.3f} vs Optimal: {results['optimal_mu']:.3f}")
    print(f"  Empirical std:  {results['empirical_std']:.3f} vs Optimal: {results['optimal_sigma']:.3f}")
    
    # Calculate KL for empirical moments
    empirical_kl = kl_divergence_continuous(results['bimodal'], 
                                           results['empirical_mean'], 
                                           results['empirical_std'])
    improvement = ((empirical_kl - results['optimal_kl']) / empirical_kl) * 100
    print(f"\nKL divergence improvement:")
    print(f"  Empirical moments KL: {empirical_kl:.6f}")
    print(f"  Optimal KL:          {results['optimal_kl']:.6f}")
    print(f"  Improvement:         {improvement:.2f}%")