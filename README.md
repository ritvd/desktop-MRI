# desktop-MRI
dMRI is the first open-source simulator built to help teaching the concepts of MR physics.

| Package     | Version |
|-------------|---------|
| Python      | 3.10.6  |
| PySimpleGUI | 4.60.4  |
| NumPy       | 1.21.5  |
| OpenCV      | 4.5.4   |

## How to Use

1. Instructions on how this simulator can be used are present in instructions.pdf in the docs folder.

2. Go to /data and run concat.py

3. Go to /src and run dMRI.py

## News
dMRI project has been selected for presentation at the Annual Conference of the Indian Society of Neuroradiology, New Delhi (2023).

## References
This project has been inspired by:
• K-space Explorer project: https://github.com/birogeri/kspace-explorer
• MRI Education: https://github.com/mriphysics/MRI_education

## Known issues
Known Issues
The brain model used in dMRI has been developed by using pixel intensity based segmentation of a 3D T1 weighted acquisition from a single subject. This may cause improper calculation of signal intensities for white matter, fat and bone. A more accurately segmented model is being developed. 

## Upcoming

We are currently working on dMRI v2.0. This advanced version of the Desktop MRI will have options for the user to choose the following:
  1. Better brain model with a more accurately segmented brain volume
  2. Option to select scanner hardware 
      - Magnet field strength 1.5 Tesla vs 3 Tesla
      - Coil configuration
  3. GUI Based slice planning
      - Tilting of plane of acquistion
      - Adjusting Field of View
      - Swapping frequency and phase encoding directions
  4. Additional Pulse sequences and Contrast mechanisms
      - Gradient echo based pulse sequences
      - Fat saturation
      - Diffusion Weighted Imaging 
      - Magnetization Transfer Imaging
      - Time of Flight MR angiography
      - Phase Contrast MR angiography
      - Arterial Spin Labelling
      - 3D Imaging
  5. Additional artefacts
      - Susceptibility Artefact
      - Chemical shift artefact
  6. Additional Anatomies 
      - Cervical spine
      - Lumbar spine
      - Knee joint
  7. Save Button to save the parameters, reconstructed images, pulse sequence diagrams and K space 



