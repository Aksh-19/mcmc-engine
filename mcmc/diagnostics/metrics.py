import numpy as np

def autocovariance(chain, max_lag=None):
    """Calculate the autocovariance of a 1D chain up to max_lag."""
    N = len(chain)
    if max_lag is None:
        max_lag = N // 2
        
    mean = np.mean(chain)
    centered = chain - mean
    
    autocov = np.zeros(max_lag)
    for k in range(max_lag):
        autocov[k] = np.dot(centered[:N-k], centered[k:]) / N
        
    return autocov

def effective_sample_size(chain):
    """Calculate the Effective Sample Size (ESS) of a 1D chain."""
    N = len(chain)
    acov = autocovariance(chain)
    
    # Autocorrelation is autocovariance normalized by variance (acov[0])
    rho = acov / acov[0]
    
    sum_rho = 0.0
    for k in range(1, len(rho)):
        if rho[k] < 0:
            break
        sum_rho += rho[k]
        
    ess = N / (1.0 + 2.0 * sum_rho)
    return ess

def r_hat(chains):
    """
    Calculate the Gelman-Rubin statistic (R-hat) for multiple chains.
    chains: 2D array of shape (M chains, N samples)
    """
    M, N = chains.shape
    
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    
    overall_mean = np.mean(chains)
    
    # Between-chain variance
    B = (N / (M - 1)) * np.sum((chain_means - overall_mean)**2)
    
    # Within-chain variance
    W = np.mean(chain_vars)
    
    # Pooled variance estimate
    var_plus = ((N - 1) / N) * W + (1 / N) * B
    
    return np.sqrt(var_plus / W)
