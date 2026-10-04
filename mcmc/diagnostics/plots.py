import matplotlib.pyplot as plt

def trace_plot(chain, title="Trace Plot", figsize=(10, 2.5)):
    """
    Plot the trace of an MCMC chain.
    
    Args:
        chain: 1D array of shape (N,) or 2D array of shape (N, D)
               where N is num_samples and D is dimensions.
        title: Title for the plot.
        figsize: Base size per dimension.
    """
    if chain.ndim == 1:
        chain = chain.reshape(-1, 1)
        
    N, D = chain.shape
    fig, axes = plt.subplots(D, 1, figsize=(figsize[0], figsize[1] * D), squeeze=False)
    
    for d in range(D):
        axes[d, 0].plot(chain[:, d], lw=0.5, alpha=0.8, color='steelblue')
        axes[d, 0].set_ylabel(f"Dimension {d}")
        axes[d, 0].grid(True, alpha=0.3)
        if d == 0:
            axes[d, 0].set_title(title)
            
    axes[-1, 0].set_xlabel("Iteration")
    plt.tight_layout()
    
    return fig, axes
