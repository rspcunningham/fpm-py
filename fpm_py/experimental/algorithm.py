import numpy as np
from dataclasses import dataclass

"""2D Fourier transform"""
_ft = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

"""2D Inverse Fourier transform"""
_ift = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x)))

"""Cropping function"""
_crop = lambda x, cen, Np: x[
    int(cen[0] - np.floor(Np[0] / 2)) : int(cen[0] - np.floor(Np[0] / 2) + Np[0]),
    int(cen[1] - np.floor(Np[1] / 2)) : int(cen[1] - np.floor(Np[1] / 2) + Np[1]),
]

@dataclass
class ReconstructionConfig:
    effective_magnification: float
    output_scale_factor: float
    pupil_0: np.ndarray
    iteration_limitter: int | callable
    optimizer: callable

def reconstruct(stack, config: ReconstructionConfig):

    # initialize values
    image_size = np.array(stack[0].image.shape)
    output_image_size = image_size * config.output_scale_factor
    du = config.effective_magnification / image_size

    object = np.zeros(output_image_size).astype(np.complex128)


def reconstruct(stack, output_scale_factor, iterations=10, effective_magnification=1.5/1.12, pupil_0=None):
    # Load pupil
    # Get initial values and hyperparameters
    if pupil_0 is None: pupil_0 = np.load('pupil.npy')
    pupil_bool = pupil_0 > 0

    # initialize values
    image_size = np.array(stack[0].image.shape)
    output_image_size = image_size * output_scale_factor
    du = effective_magnification / image_size

    O = np.zeros(output_image_size).astype(np.complex128)
    center_image = stack.pop(0)
    Os = _ft(np.sqrt(center_image.image))  

    fourier_center = np.round(output_image_size / 2).astype(np.int32)
    n1 = (fourier_center - np.floor(image_size / 2)).astype(np.int32) # Bottom left
    n2 = (n1 + image_size - 1).astype(np.int32) # Top right

    O[n1[0]-1:n2[0], n1[1]-1:n2[1]] = Os * pupil_0 # initialize the object with the first image

    P = pupil_0  # This just looks like a binary mask of a circle for now, but will be updated with precomputed pupil function
    
    for i in range(iterations):
        for j, data in enumerate(stack):
            
            center = fourier_center + np.round(data.k_vector / du).astype(np.int32)
            I_measured = data.image

            # Compute estimated exit wave (step 5)
            psi_fourier = _crop(O, center, image_size) * P

            # Propagate to spatial domain (step 6)
            psi_spatial = _ift(psi_fourier)

            # Spatial optimization (step 7, 8)
            I_estimated = np.abs(psi_spatial) ** 2
            
            psi_fourier_prime = _ft(
                np.sqrt(I_measured) * psi_spatial / (np.sqrt(I_estimated) + np.finfo(np.float64).eps)
            )

            # Fourier optimization (step 9)
            delta_psi = psi_fourier_prime - psi_fourier
            O,P = gradient_descent(O, P, delta_psi, center, pupil_bool)

        print(f"Completed Iteration {i + 1}")

    o = _ift(O)
    return o # Note that the return value is the complex field, not the intensity; take the absolute value to get the intensity
