import numpy as np


def objective_function(a: np.ndarray, M: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the total reconstruction error for a single pixel.
    Constrained optimization, minimizing ||Ma - y||^2, subject to sum(a)=1 and a>=0.

    Args:
        a: Array of fractional abundances (flattened over the spatial dimension), of shape (n_pixels, nb_endmembers).
        M: Endmember matrix, of shape (n_wavelengths, nb_endmembers).
        y: Observed spectral signature (flattened over the spatial dimension), of shape (n_pixels, n_wavelengths).

    Returns:
        Scalar sum of squared residuals.
    """
    y_hat = a @ M.T
    residuals = y_hat - y
    out = np.sum(residuals**2)
    return out


def generate_abundance_map(height: int, width: int, nb_endmembers: int) -> np.ndarray:
    """
    Generate a synthetic abundance map where each pixel sums to 1.

    Args:
        height: Number of rows.
        width: Number of columns.
        nb_endmembers: Number of spectral constituents.

    Returns:
        Abundance map of shape (height, width, nb_endmembers).
    """
    # Generate positive values uniformly at random, ensuring non-negativity (ANC)
    abundance_map = np.random.rand(height, width, nb_endmembers)

    # Enforce the sum-to-one constraint (ASC)
    abundance_map /= np.sum(abundance_map, axis=2, keepdims=True)

    return abundance_map
