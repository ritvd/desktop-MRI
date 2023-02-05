# desktop-MRI
dMRI is the first open-source simulator built to help teaching the concepts of MR physics.

| Package     | Version |
|-------------|---------|
| Python      | 3.10.6  |
| PySimpleGUI | 4.60.4  |
| NumPy       | 1.21.5  |
| OpenCV      | 4.5.4   |

## Layout of dMRI

dMRI graphical user interface is organized into five panels.

**A. User Input Panel:**
Displays all the important user adjustable parameters that are available on a clinical MR scanner 
The parameters are organized into four tabs:  
  1. Localization parameters: Parameters that determine the location and dimensions of the reconstructed slices.
  2. Acquisition parameters: Parameters that determine the location and dimensions of the voxels constituting the reconstructed slices, k-Space trajectory employed for       data collection and number of signal averages.
  3. Contrast parameters: Parameters that determine the voxel signal intensities in reconstructed slices. These include the pulse sequence and parameters such as Flip      angle, TR, TE, Inversion time and bandwidth.
  4. Additional parameters: Additional k-Space strategy related parameters, choice of artifacts and image filters that can be introduced into the reconstructed image         are listed in this tab. It also contains sliders to control the magnitude or location of some of these parameters see ‘Steps of dMRI image reconstruction’ below       for details of the parameters in the user input panel.
  
**B. Output Panel:**
