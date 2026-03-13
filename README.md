# Spectral unmixing

![banner](https://github.com/user-attachments/assets/8d4ea8c8-ede4-4b31-b1d6-b510f361a55a)

## Definition

In the simplest case, given a medium containing $p$ different materials (endmembers) observed under $L$ spectral bands (wavelengths), the observed reflectance at each pixel can be modeled by:

$$\displaystyle \mathbf{y} = \mathbf{E}\mathbf{a} + \mathbf{n}$$

with:

- $\mathbf{y}:$ Observed pixel vector, of shape $(L,)$.
- $\mathbf{E}:$ Endmember matrix, of shape $(L, p)$, where each column is the spectral signature of a material.
- $\mathbf{a}:$ Fractional abundance vector, of shape $(p,)$.
- $\mathbf{n}:$ Additive measurement noise vector, of shape $(L,)$.

**Spectral unmixing** is the process of decomposing the measured spectrum of a mixed pixel into a collection of constituent spectral signatures (endmembers) and their corresponding proportions (abundances). While the forward process (mixing) combines materials into a single observation (reflectance), unmixing seeks to solve the inverse problem to reveal the underlying composition of the scene.

To solve this inverse problem, one typically looks for the abundance vector $\mathbf{\hat{a}}$ that minimizes the discrepancy between the model and the observation. This approach is framed as a least squares optimization:

$$\displaystyle \mathbf{\hat{a}} =\arg \min_{\mathbf{a}} \|\mathbf{y} - \mathbf{E}\mathbf{a}\|_2^2$$

In more complex scenarri, several degrading phenomena (e.g., spectral coloring, chemical interaction, multiple scattering, shadowing, occlusion, fluence attenuation) make the inverse problem nonlinear, unstable, and/or ill-posed. In these circumstances, the unmixing task is markedly more challenging, and more advanced algorithms than least squares optimization must be used.

## Selected bibliography

**N Keshava and JF Mustard (2002).** *"Spectral unmixing."* IEEE signal processing magazine.

**DC Heinz and CI Chang (2001).** *"Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery."* IEEE transactions on geoscience and remote sensing.

**N Dobigeon, JY Tourneret, C Richard, JCM Bermudez, S McLaughlin, AO Hero (2013).** *"Nonlinear unmixing of hyperspectral images: Models and algorithms."* IEEE Signal processing magazine.

**W Fan, B Hu, J Miller, M Li (2009).** *"Comparative study between a new nonlinear model and common linear model for analysing laboratory simulated‐forest hyperspectral data."* International Journal of Remote Sensing.

**JMP Nascimento and JM Bioucas-Dias (2009).** *"Nonlinear mixture model for hyperspectral unmixing."* In Image and Signal Processing for Remote Sensing XV.

**R Heylen, M Parente, P Gader (2014).** *"A review of nonlinear hyperspectral unmixing methods."* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

**JM Bioucas-Dias and A Plaza (2010).** *"Hyperspectral unmixing: Geometrical, statistical, and sparse regression-based approaches."* In Image and signal processing for remote sensing XVI.

**B Cox, JG Laufer, SR Arridge, PC Beard (2012).** *"Quantitative spectroscopic photoacoustic imaging: a review."* Journal of biomedical optics.

**B Palsson, J Sigurdsson, JR Sveinsson, MO Ulfarsson (2018).** *"Hyperspectral unmixing using a neural network autoencoder."* IEEE Access.

**L Gao, Z Han, D Hong, B Zhang (2021).** *"CyCU-Net: Cycle-consistency unmixing network by learning cascaded autoencoders."* IEEE Transactions on Geoscience and Remote Sensing.

**M Zhao, M Wang, J Chen, S Rahardja (2021).** *"Hyperspectral unmixing for additive nonlinear models with a 3-D-CNN autoencoder network."* IEEE Transactions on Geoscience and Remote Sensing.

**M Wang, M Zhao, J Chen, S Rahardja (2019)."** *"Nonlinear unmixing of hyperspectral data via deep autoencoder networks."* IEEE Geoscience and Remote Sensing Letters.

**K Mantripragada and FZ Qureshi (2024).** *"Hyperspectral pixel unmixing with latent Dirichlet variational autoencoder."* IEEE Transactions on Geoscience and Remote Sensing.

## Installation & Setup

**1. Install Pipenv (if not already installed):**

```sh
pip install --user pipenv
```

**2. Initialize environment:**

```sh
pipenv install --dev --python 3.12
```

**3. Activate environment:**

```sh
pipenv shell
```
