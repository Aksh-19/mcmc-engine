class Distribution:
    """Base class for all probability distributions."""

    def log_prob(self, x):
        """Return log π(x). Subclasses must implement this."""
        raise NotImplementedError

    def __call__(self, x):
        """Shortcut: dist(x) instead of dist.log_prob(x)."""
        return self.log_prob(x)    # needs 'return' and 'self.'