import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr, linregress


def plot_endmember_spectral_signatures(
    endmember_matrix: np.ndarray,
    wavelengths: np.ndarray,
    labels: list
) -> None:

    nb_endmembers = endmember_matrix.shape[1]

    plt.figure(figsize=(8, 4))
    for e in range(nb_endmembers):
        plt.plot(wavelengths, endmember_matrix[:, e], marker="o", label=labels[e], linewidth=2)
    plt.title("Endmember spectral signatures ($M$)", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance (a.u.)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_abundance_maps(abundance_map, abundance_map_estimated, abundance_map_error, labels) -> None:

    map_height, map_width, nb_endmembers = abundance_map.shape

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    column_names = ["Ground Truth", "Estimated", "Error"]
    data = [abundance_map, abundance_map_estimated, abundance_map_error]

    plt.suptitle("Abundance maps", fontsize=14)

    for row in range(nb_endmembers):
        for col in range(3):
            ax = axes[row, col]
            img_data = data[col][:, :, row]

            if col < 2:
                vmin, vmax = 0, 1
                cmap = "viridis"
            else:
                limit = np.max(np.abs(img_data))
                vmin, vmax = -limit, limit
                cmap = "RdBu_r"

            im = ax.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax)
            nice_colorbar(im=im, ax=ax)

            # Add white text annotations for each cell
            for i in range(map_height):
                for j in range(map_width):
                    ax.text(j, i, f"{img_data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

            if row == 0: ax.set_title(column_names[col], fontsize=12)
            if col == 0: ax.set_ylabel(labels[row], fontsize=12)
            ax.set_xticks([]);
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_linear_regression_and_bland_altman(abundance_map, abundance_map_estimated):

    y_gt = abundance_map.flatten()
    y_hat = abundance_map_estimated.flatten()

    r, _ = pearsonr(y_gt, y_hat)
    slope, intercept, _, _, _ = linregress(y_gt, y_hat)

    difference = y_hat - y_gt
    mean_val = (y_hat + y_gt) / 2
    difference_mean = np.mean(difference)
    difference_std = np.std(difference)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot and regression
    ax1.scatter(y_gt, y_hat, alpha=0.5, color="darkorange", s=20)
    ax1.plot(y_gt, slope * y_gt + intercept, linewidth=2, color="steelblue", label=f"Fit: y={slope:.2f}x + {intercept:.2f}")
    ax1.plot([0, 1], [0, 1], color="black", linewidth=2, linestyle="--", label="Identity (y=x)")
    ax1.set_xlabel("True abundance", fontsize=12)
    ax1.set_ylabel("Estimated abundance", fontsize=12)
    ax1.set_title(f"Scatter plot (Pearson r = {r:.3f})", fontsize=14)
    ax1.legend()

    # Bland-Altman plot
    ax2.scatter(mean_val, difference, alpha=0.5, color="darkorange", s=20)
    ax2.axhline(difference_mean, linewidth=2, color="steelblue", linestyle="-", label=f"Mean difference: {difference_mean:.3f}")
    ax2.axhline(difference_mean + 1.96*difference_std, linewidth=2, color="steelblue", linestyle="--", label=f"+1.96 difference_std: {difference_mean + 1.96*difference_std:.3f}")
    ax2.axhline(difference_mean - 1.96*difference_std, linewidth=2, color="steelblue", linestyle="--", label=f"-1.96 difference_std: {difference_mean - 1.96*difference_std:.3f}")
    ax2.set_xlabel("Average Abundance", fontsize=12)
    ax2.set_ylabel("Signed difference (Estimated - Ground truth)", fontsize=12)
    ax2.set_title("Bland-Altman plot", fontsize=14)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def nice_colorbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    return cbar
