import numpy as np

_ft = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
_ift = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x)))
_crop = lambda x, cen, Np: x[
    int(cen[0] - np.floor(Np[0] / 2)) : int(cen[0] - np.floor(Np[0] / 2) + Np[0]),
    int(cen[1] - np.floor(Np[1] / 2)) : int(cen[1] - np.floor(Np[1] / 2) + Np[1]),
]

def gradient_descent(object, pupil, pupil_binary, dpsi, center):
    alpha_o = 1
    mu_o = 1
    alpha_p = 1
    mu_p = 1

    pupil_dims = np.asarray(np.shape(pupil))
    
    # operator to put pupil at proper location at the object plane
    n1 = (center - np.floor(pupil_dims / 2)).astype("int")
    n2 = (n1 + pupil_dims - 1).astype("int")

    slices = (slice(n1[0], n2[0]), slice(n1[1], n2[1]))
    
    # selects only the region of interst for O, leaves everything else alone
    object_cropped = object[slices]
        
    # Update the object with the corrected values
    object[slices] = object_cropped + (
        alpha_o * np.abs(pupil) * np.conj(pupil) * dpsi
    ) / (np.max(np.abs(pupil)) * (np.abs(pupil) ** 2 + mu_o))
    
    # Update the pupil with the correction term
    pupil += (
        alpha_p * np.abs(object_cropped) * np.conj(object_cropped) * dpsi
    ) / (np.max(np.abs(object_cropped)) * (np.abs(object_cropped) ** 2 + mu_p)) * pupil_binary

    return object, pupil

def reconstruct(stack, output_scale_factor):
    # load pupil
    # get initial values and hyperparameters


    # this is the actual algorithm
    # stack is a list of Data objects
    iterations = 10
    effective_magnification = 1.5/1.12

    pupil_0 = np.load('./src/pipeline/pupil.npy')
    pupil_bool = pupil_0 > 0

    image_size = np.array(stack[0].image.shape)
    output_image_size = image_size * output_scale_factor

    du = effective_magnification / image_size

    O = np.zeros(output_image_size).astype(np.complex128)
    center_image = stack.pop(0)
    Os = _ft(np.sqrt(center_image.image))  

    fourier_center = np.round(output_image_size / 2).astype(np.int32)
    n1 = (fourier_center - np.floor(image_size / 2)).astype(np.int32) #bottom left
    n2 = (n1 + image_size - 1).astype(np.int32) #top right

    O[n1[0]-1:n2[0], n1[1]-1:n2[1]] = Os * pupil_0

    P = pupil_0  # this just looks like a binary mask of a circle for now, but will be updated with precomputed pupil function
    
    for i in range(iterations):
        for j, data in enumerate(stack):
            
            center = fourier_center + np.round(data.k_vector / du).astype(np.int32)
            I_measured = data.image

            # compute estimated exit wave (step 5)
            psi_fourier = _crop(O, center, image_size) * P

            # propagate to spatial domain (step 6)
            psi_spatial = _ift(psi_fourier)

            # spatial optimization (step 7, 8)
            I_estimated = np.abs(psi_spatial) ** 2
            
            psi_fourier_prime = _ft(
                np.sqrt(I_measured) * psi_spatial / (np.sqrt(I_estimated) + np.finfo(np.float64).eps)
            )

            # fourier optimization (step 9)
            delta_psi = psi_fourier_prime - psi_fourier
            O,P = gradient_descent(O, P, delta_psi, center, pupil_bool)

        print(f"Completed Iteration {i + 1}")

    o = _ift(O)
    return o # note that the return value is the complex field, not the intensity; take the absolute value to get the intensity

