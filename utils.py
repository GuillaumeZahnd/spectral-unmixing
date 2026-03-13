import numpy as np
from scipy.optimize import nnls


def objective_function(a: np.ndarray, M: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the total reconstruction error for a single pixel.
    Constrained optimization, minimizing ||Ma - y||^2, subject to sum(a)=1 and a>=0.

    Args:
        a: Array of fractional abundances, of shape (nb_endmembers,).
        M: Endmember matrix, of shape (n_wavelengths, nb_endmembers).
        y: Observed spectral signature, of shape (n_wavelengths,).

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


def gaussian_function(x: float | np.ndarray, a: float, b: float, c: float) -> float | np.ndarray:
    """
    Evaluate a Gaussian function.

    Args:
        x: Input value(s), can either be a scalar or a NumPy array.
        a: Amplitude.
        b: Center.
        c: Standard deviation.

    Returns:
        Gaussian function evaluated at x.
    """
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def compute_rmse(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute the root mean square error between a prediction and the ground truth."""
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


def compute_fcls(spectrum: np.ndarray, M: np.ndarray, lambda_penalty: float = 10) -> np.ndarray:
    """
    Apply the Fully Constrained Least Squares (FCLS) algorithm for hyperspectral unmixing.

    Args:
        spectrum: Observed spectral signature, of shape (n_wavelengths,).
        M: Endmember matrix, of shape (n_wavelengths, nb_endmembers).
        lambda_penalty: Parameter to control how strongly the sum-to-one constraint is enforced.
    Returns:
        Relative abundances (estimated fraction of each material in the pixel), of shape (nb_endmembers,).
    """

    if spectrum.shape[0] != M.shape[0]:
        raise ValueError("Spectrum and endmember matrix must have the same number of bands.")

    eps = 1e-9
    nb_endmembers = M.shape[1]
    endmember_matrix_augmented = np.vstack([M, lambda_penalty * np.ones((1, nb_endmembers))])
    spectrum_augmented = np.append(spectrum, lambda_penalty)
    abundances, _ = nnls(endmember_matrix_augmented, spectrum_augmented)
    abundances /= abundances.sum() + eps
    return abundances
