# Fourier Ptychography Experiment Ideas

## 1) k-noise --> `k-space-exploration`

 - define k vectors
 - generate simulated captures with those k vectors
 - add random noise to the k vector positions
 - reconstruct with those noisy k vectors, learning the real k positions
 

## 2) non-point sources

 - light sources are not points. can we model that more effectively/accurately?
 - plane wave vs spherical wave
 - does this impact the phase ramps we apply
 
## 3) k-vector positions

 - how do different patterns in k-space impact the quality of reconstruction?
 - can we learn the optimal k-space pattern for n vectors?
 
## 4) different images

 - currently we are just using the bar pattern for everything
 - how do different images impact the quality of reconstruction?
 - can we use real images? 
 
## 5) pupil parameterization

 - learning the pupil as a raw HxW tensor doesnt work perfectly, there are obvious issues
 - can we solve these by parameterizing it better?
 - Zernike polynomials? LoRA?

## 6) abherrations

 - abbherations are changes to the pupil function
 - can we model common abherrations in the forward pass, ie synthesize them?
 - if we can synthesize then, can we learn to correct for them
 
## 7) nonuniform pupil function

 - since images at different k-vectors take different optical paths through the lens, they have different pupil functions
 - fov-splitting is supposed to help manage this
 - can we maximally fov-split, ie a new pupil for each pixel in the object?
 - logically, these 'subpupils' will all be related in some way. can we model that relationship using a shared parameterization?
 
## 8) FPNet

 - can we train a neural net to do reconstruction in fewer iterations? what would that look like
 - what other things can we train a model to model.
