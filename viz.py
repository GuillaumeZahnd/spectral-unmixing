import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def nice_colorbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    return cbar
