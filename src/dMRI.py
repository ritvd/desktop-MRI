#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 23:13:33 2023

@author: Rithika Varma

"""

import numpy as np
import cv2
import PySimpleGUI as sg
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import math

sg.change_look_and_feel('DefaultNoMoreNagging')
brain_model=np.load('data/brain_vol.npy')

brain_model[brain_model < 10] = 1                           # 1 = Air 
brain_model[(brain_model > 9) & (brain_model < 80)] = 2       # 2 = CSF        
brain_model[(brain_model > 79) & (brain_model < 151)] = 3     # 3 = GM
brain_model[(brain_model > 150) & (brain_model < 181)] = 4    # 4 = WM 
brain_model[(brain_model > 180)] = 5                        # 5 = Fat

# DECLARING THE TISSUE TYPE, T1 VALUE, T2* VALUE, PD VALUE

parameters = {
1: ["Air", 1, 1, 1, 1, 0, 0, 0, 0, 0],
2: ["CSF", 2, 4300, 1200, 91, 1, 63, 129, 235, 0],
3: ["Grey Matter", 3, 1023, 69, 50, 0.81, 214, 112, 49, 0],
4: ["White Matter", 4, 653, 67, 45, 0.59, 129, 214, 49, 0],
5: ["Fat", 5, 500, 85, 45, 1, 255, 247, 0, 0],
}



# DECLARING THE STATIC VARIABLES

sensetimecorr= {1: [1], 2: [0.5], 3: [0.36], 4: [0.27], 5: [0.19]}
image=0
psdgraph=[]
message1_text=""
calc_snr_message="Calculated SNR            : \nCalculated Scan Time :"  
param_message=""
snr_value=0
sense_snr_corr = 1
sense_time_corr = 1 
sensetimecorrlist= {1: [1], 2: [0.5], 3: [0.36], 4: [0.27], 5: [0.19]}
result_values = {} 

# DECLARING THE DEFAULT VALUES FOR THE SLIDERS
key_val_dict = { 5:1, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 
                15:1, 16:0, 17:0, 18:1, 19:0, 20:0, 21:10, 22:4, 23:1, 24:0, 25:0, 26:-12, 
                27:1, 28:0, 29:260, 30:100, 31:256, 32:256, 
                33:1, 34:0, 35:0, 36:0, 37:0, 
                38:90, 39:500, 40:12, 41:0, 42:32, 43:0, 44:0, 45:0, 46:0, 47:1, 48:0, 49:0, 
                50:1, 51:0, 52:0, 53:0, 54:0, 55:0, 56:0, 
                57:0, 58:0, 59:0, 58:0, 59:0, 60:0, 61:0, 62:0, 63:0, 64:0, 65:0, 66:0, 67:50, 68:50, 
                69:0, 70:0, 71:0, 72:0, 73:0, 74:0, 75:0} 

# INITIATING THE DYNAMIC VARIABLES

def initialize_clean_arrays():
    global all_slices
    global slice_comp
    global slice_comp_padded
    global image_data
    global mask_data
    global relax_array
    global psd_array
    global traj_array
    global traj_array_text
    global zoom_data
    
    all_slices=np.zeros((512,512,32),dtype=np.uint8)
    slice_comp=np.zeros((512,512),dtype=np.uint8)
    slice_comp_padded=np.zeros((1024,1024),dtype=np.uint8)
    image_data=np.zeros((512,512,32),dtype=np.uint8)
    mask_data=np.zeros((512,512,32),dtype=np.uint8)
    relax_array=np.zeros((512,512,3),dtype=np.uint8)
    psd_array=np.zeros((512,512,3),dtype=np.uint8)
    traj_array=np.zeros((512,512,3),dtype=np.uint8)
    traj_array_text=np.zeros((512,512,3),dtype=np.uint8)
    zoom_data=np.zeros((512,512,32),dtype=np.uint8)

# RESET THE SLIDERS WITH RESET BUTTON
def reset_values():
    pseqcse=key_val_dict[5]; pseqfse=key_val_dict[6]; etlvalue=1; pseqmir=key_val_dict[7]; pseqpsir=key_val_dict[10]; pseqspgr=key_val_dict[11];
    fastmodenone=key_val_dict[15]; fastmodesense=key_val_dict[16]; sensefactor=1; fastmodecsense=key_val_dict[17]; csensefactor=1;
    acqplaneax=key_val_dict[18]; acqplanecor=key_val_dict[19]; acqplanesag=key_val_dict[20]; noofslices=key_val_dict[21]; slicethkness=key_val_dict[22]; slicegap=key_val_dict[23]; rloffset=key_val_dict[24]; apoffset=key_val_dict[25]; hfoffset=key_val_dict[26];
    fovfreq=key_val_dict[29];phasewrap=key_val_dict[27]; nophasewrap=key_val_dict[28]; acqfreq=key_val_dict[31]; acqphase=key_val_dict[32];
    acqmodecart=key_val_dict[33]; acqmodeepi=key_val_dict[34]; epishots=1; acqmoderad=key_val_dict[35]; radialspokes=100; acqmodeprop=key_val_dict[36]; propblades=20; acqmodespir=key_val_dict[37]; spirinterleaves=1;
    faval=key_val_dict[38]; trval=key_val_dict[39]; teval=key_val_dict[40]; tival=key_val_dict[41]; bwval=key_val_dict[42]; nexval=key_val_dict[46]; filtsmooth=key_val_dict[47]; filtnormal=key_val_dict[48]; filtsharp=key_val_dict[49];
    kspacefa=key_val_dict[50]; kspacepf=key_val_dict[51]; kspacepe=key_val_dict[52]; kspacec=key_val_dict[53]; kspacep=key_val_dict[54]; kspaceu=key_val_dict[55]; kspacez=key_val_dict[56];
    iqmotion=key_val_dict[58]; iqspike=key_val_dict[60]; iqzipper=key_val_dict[62]; iqnyghosts=key_val_dict[63]; iqrfclip=key_val_dict[64]; iqdcoffset=key_val_dict[65]; iqrfshade=key_val_dict[66]; kspsliderval1=key_val_dict[67]; kspsliderval2=key_val_dict[68];
    currslice=5; brightness=50; contrast=50;

    window['-PSEQCSE-'](pseqcse), window['-PSEQFSE-'](pseqfse), window['-ETLVALUE-'](etlvalue), window['-PSEQMIR-'](pseqmir), window['-PSEQPSIR-'](pseqpsir), window['-PSEQSPGR-'](pseqspgr),
    window['-FASTMODENONE-'](fastmodenone), window['-FASTMODESENSE-'](fastmodesense), window['-SENSEFACTOR-'](sensefactor), window['-FASTMODECSENSE-'](fastmodecsense), window['-CSENSEFACTOR-'](csensefactor),
    window['-ACQPLANEAX-'](acqplaneax), window['-ACQPLANECOR-'](acqplanecor), window['-ACQPLANESAG-'](acqplanesag), window['-NOOFSLICES-'](noofslices), window['-SLICETHKNESS-'](slicethkness), window['-SLICEGAP-'](slicegap), window['-RLOFFSET-'](rloffset), window['-APOFFSET-'](apoffset), window['-HFOFFSET-'](hfoffset),
    window['-FOVFREQ-'](fovfreq), window['-PHASEWRAP-'](phasewrap), window['-NOPHASEWRAP-'](nophasewrap), window['-ACQFREQ-'](acqfreq), window['-ACQPHASE-'](acqphase),
    window['-ACQMODECART-'](acqmodecart), window['-ACQMODEEPI-'](acqmodeepi), window['-EPISHOTS-'](epishots), window['-ACQMODERAD-'](acqmoderad), window['-RADIALSPOKES-'](radialspokes), window['-ACQMODEPROP-'](acqmodeprop), window['-PROPBLADES-'](propblades), window['-ACQMODESPIR-'](acqmodespir), window['-SPIRINTERLEAVES-'](spirinterleaves),
    window['-FAVAL-'](faval), window['-TRVAL-'](trval), window['-TEVAL-'](teval), window['-TIVAL-'](tival), window['-BWVAL-'](bwval), window['-NEXVAL-'](nexval), window['-FILTSMOOTH-'](filtsmooth), window['-FILTNORMAL-'](filtnormal), window['-FILTSHARP-'](filtsharp),
    window['-KSPACEFA-'](kspacefa), window['-KSPACEPF-'](kspacepf), window['-KSPACEPE-'](kspacepe), window['-KSPACEC-'](kspacec), window['-KSPACEP-'](kspacep), window['-KSPACEU-'](kspaceu), window['-KSPACEZ-'](kspacez),
    window['-IQMOTION-'](iqmotion), window['-IQSPIKE-'](iqspike), window['-IQZIPPER-'](iqzipper), window['-IQNYGHOSTS-'](iqnyghosts), window['-IQRFCLIP-'](iqrfclip), window['-IQDCOFFSET-'](iqdcoffset), window['-IQRFSHADE-'](iqrfshade), window['-KSPSLIDERVAL1-'](kspsliderval1), window['-KSPSLIDERVAL2-'](kspsliderval2),
    window['-CURRSLICE-'](currslice), window['-BRIGHTNESS-'](brightness), window['-CONTRAST-'](contrast)

# DYNAMICALLY UPDATE IMAGE BASED ON CONTRAST AND BRIGHTNESS AS WELL AS UPDATE THE MESSAGES FIELD
def update_image(noofslices,imagenumber,brightness,contrast, pseqcse, pseqfse, pseqmir, pseqpsir, pseqspgr, kspacefa, kspacepf, kspacepe, kspacec, kspacep, kspaceu, kspacez, iqmotion, iqspike, iqzipper, iqnyghosts, iqrfclip, iqdcoffset, iqrfshade, acqmodecart, acqmodeprop, acqmodeepi, filtsmooth, filtnormal, filtsharp, imageinvert, zoomh, zoomv, snr_value):
    
    global all_slices
    global mask_data
    global relax_array
    global psd_array
    global traj_array
    global zoom_data

    zoom_extract=np.zeros((200,200),dtype=np.uint8)
    magn_recon_image=np.zeros((512,512),dtype=np.uint8)
    zoomh=int(zoomh)
    zoomv=int(zoomv)
    curr_image_array = all_slices[:,:,int(imagenumber-1)]

    if snr_value ==0:
        noise_val = 0
        noise_data=np.load('data/nonoise.npy')
    else:
        noise_val = int((snr_value/10)+1)
        if noise_val > 10:
            noise_val = 10
        noise_data=np.load('data/'+str(noise_val)+'.npy')

    curr_noised_image = curr_image_array + noise_data

    image_generated=Image.fromarray(curr_noised_image)
    image_generated= image_generated.convert('L')
    enhancer = ImageEnhance.Brightness(image_generated)
    image = enhancer.enhance(brightness/50)
    enhancer = ImageEnhance.Contrast(image) 
    image = enhancer.enhance(contrast/50)

    if filtsmooth == 1:
        image = image.filter(ImageFilter.SMOOTH)
    if filtnormal== 1:
        image = image
    if filtsharp == 1:
        image = image.filter(ImageFilter.SHARPEN)

    with BytesIO() as output:
        image.save(output, format="PNG")
        content_image = output.getvalue()
        window['-CURRIMAGE-'].update(data=content_image)
        window['-CURRSLICE-'].update(range = (1, noofslices))

    kspace_generated=Image.fromarray(mask_data[:,:,int(imagenumber-1)])
    kspace_generated= kspace_generated.convert('L')
    with BytesIO() as output2:
        kspace_generated.save(output2, format="PNG")
        content_kspace = output2.getvalue()
        window['-CURRMASK-'].update(data=content_kspace)

    relaxation_curves=Image.fromarray(relax_array)
    relaxation_curves= relaxation_curves.convert('RGB')
    with BytesIO() as output3:
        relaxation_curves.save(output3, format="PNG")
        relaxation_image= output3.getvalue()
        window['-RELAXATIONS-'].update(data=relaxation_image)

    psd_curves=Image.fromarray(psd_array)
    psd_curves= psd_curves.convert('RGB')
    with BytesIO() as output4:
        psd_curves.save(output4, format="PNG")
        psd_image= output4.getvalue()
        window['-PSDGRAPH-'].update(data=psd_image)

    kspace_traj=Image.fromarray(traj_array)
    kspace_traj= kspace_traj.convert('RGB')
    with BytesIO() as output5:
        kspace_traj.save(output5, format="PNG")
        traj_kspace = output5.getvalue()
        window['-KSPACETRAJ-'].update(data=traj_kspace)

    zoom_centre_h = int(255-zoomv)
    zoom_centre_v = int(255+zoomh)
    zoom_startx = zoom_centre_h -100; zoom_endx = zoom_centre_h +100; zoom_starty = zoom_centre_v - 100 ; zoom_endy = zoom_centre_v + 100 
    zoom_extract = curr_image_array[zoom_startx:zoom_endx, zoom_starty:zoom_endy]
    noise_extract = noise_data[zoom_startx:zoom_endx, zoom_starty:zoom_endy]
    
    magn_recon_image = cv2.resize(zoom_extract, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    magn_noise_extract = cv2.resize(noise_extract, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

    magn_noised_image = magn_recon_image + magn_noise_extract
    zoom_generated=Image.fromarray(magn_noised_image[:,:])
    zoom_generated= zoom_generated.convert('L')
    enhancer = ImageEnhance.Brightness(zoom_generated)
    image_zoom = enhancer.enhance(brightness/50)
    enhancer = ImageEnhance.Contrast(image_zoom) 
    image_zoom = enhancer.enhance(contrast/50)

    if filtsmooth == 1:
        image_zoom = image_zoom.filter(ImageFilter.SMOOTH)
    if filtnormal== 1:
        image_zoom = image_zoom
    if filtsharp == 1:
        image_zoom = image_zoom.filter(ImageFilter.SHARPEN)

    with BytesIO() as output6:
        image_zoom.save(output6, format="PNG")
        content_zoom = output6.getvalue()
        window['-ZOOMIMAGE-'].update(data=content_zoom)

    # CHANGE SLIDER LIMITS BASED ON PULSE SEQUENCE CHOSEN
    if pseqcse==1:
        window['-FAVAL-'].update(range = (90, 90))
        window['-TIVAL-'].update(range = (0, 0))
        window['-TRVAL-'].update(range = (0, 10000))
        window['-TEVAL-'].update(range = (0, 200))

    if pseqfse==1:
        window['-FAVAL-'].update(range = (90, 90))
        window['-TIVAL-'].update(range = (0, 0))
        window['-TRVAL-'].update(range = (0, 10000))
        window['-TEVAL-'].update(range = (0, 200))

    if pseqmir==1:
        window['-FAVAL-'].update(range = (90, 90))
        window['-TIVAL-'].update(range = (0, 3000))
        window['-TRVAL-'].update(range = (0, 10000))
        window['-TEVAL-'].update(range = (0, 200))

    if pseqpsir==1:
        window['-FAVAL-'].update(range = (90, 90))
        window['-TIVAL-'].update(range = (0, 3000))
        window['-TRVAL-'].update(range = (0, 10000))
        window['-TEVAL-'].update(range = (0, 200))
        
    if pseqspgr==1:
        window['-FAVAL-'].update(range = (0, 90))
        window['-TIVAL-'].update(range = (0, 0))
        window['-TRVAL-'].update(range = (0, 1000))
        window['-TEVAL-'].update(range = (0, 50))


    # USER GUIDANCE DISPLAY MESSAGES UPDATED HERE
    if kspacefa==1:
        message1_text=""
        window['-MESSAGE1-'].update(message1_text)

    if kspacepf==1:
        message1_text="Slider A controls the % of phase encoding steps acquired. \nRemember: Atleast 51% phase encodes are needed for image reconstruction"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if kspacepe==1:
        message1_text="Slider A controls the % of frequency encoding steps acquired. \nRemember: Atleast 51% frequency encodes are needed for image \nreconstruction"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if kspacec==1:
        message1_text="Slider A controls the % of central kSpace used for reconstruction. \nThis acts as a low pass filter"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if kspacep==1:
        message1_text="Slider controls the % of central kSpace removed before reconstruction. \nThis acts as a high pass filter"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if kspaceu==1:
        message1_text="Slider A controls undersampling factor \n2 indicates that alternate lines of kSpace are acquired"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (2, 5))

    if kspacez==1:
        message1_text="Slider A controls undersampling factor \nUndersampling factor of 2 indicates that alternate lines of kSpace \nare acquired and rest of kSpace is zero filled"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (2, 5))
      
    if iqmotion==1:
        message1_text="Slider A controls quantum of motion & Slider B controls when it \noccurs during the data acquisition"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if iqspike==1:
        message1_text="Slider A & B control location of the kSpace spike along the Phase and \nFrequency axes \n 50 indicates centre of kSpace, while 0 and 100 indicate periphery of \nkSpace along that axis"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL2-'].update(disabled=False)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))
        window['-KSPSLIDERVAL2-'].update(range = (1, 100))
        
    if iqnyghosts==1 and acqmodeepi==1:
        message1_text="Slider A changes the magnitude of alternate lines of the kSpace"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 50))

    if iqnyghosts==1 and acqmodeepi!=1:
        message1_text="Nyquist ghosts are seen in EPI acquisitions which have a zig zag trajectory \nin the kSpace!"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if iqrfclip==1:
        message1_text="Slider A controls receiver gain."
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if iqrfshade==1:
        message1_text="Slider A controls the degree of RF loss due to faulty receiver coil"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if iqzipper==1:
        message1_text="Slider A controls the frequency of the aberrant RF waveform"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))

    if iqdcoffset==1:
        message1_text="Note the focus of increased signal at the centre of the image"
        window['-MESSAGE1-'].update(message1_text)

    if iqrfshade==1:
        message1_text="Slider B controls the malfunctioning coil element"
        window['-MESSAGE1-'].update(message1_text)
        window['-KSPSLIDERVAL1-'].update(range = (1, 100))
        window['-KSPSLIDERVAL2-'].update(range = (1, 4))

# K SPACE GENERATION FROM IMAGE
def gen_k_space(image_data):
    dft = np.fft.fft2(image_data)
    dft_shift = np.fft.fftshift(dft)
    dft_shift = np.roll(dft_shift, 1, axis=(0))

    mag_spectrum = 20 * np.log(np.abs(dft_shift))
    return mag_spectrum, dft_shift

# IMAGE RECONSTRUCTION FROM K SPACE 
def reconstruct(shift):
    ifshift = np.fft.ifftshift(shift)
    reconst = np.fft.ifft2(ifshift)
    reconst = np.abs(reconst)
    return reconst

# TRANSPOSING THE BRAIN MODEL FOR SLICING INTO AXIAL, CORONAL OR SAGITTAL PLANE
def axial2axial (brain_model):
    global brain_vol
    brain_vol = np.pad(brain_model, ((0,0), (0,0), (256, 256)), constant_values=0)
    return(brain_vol)

def axial2coronal (brain_model):
    global brain_vol
    brain_model = np.transpose(brain_model, (2, 1, 0))
    brain_vol = np.pad(brain_model, ((0,0), (0,0), (256, 256)), constant_values=0)
    return(brain_vol)

def axial2sagittal (brain_model):
    global brain_vol
    brain_model = np.transpose(brain_model, (2, 0, 1))
    brain_vol = np.pad(brain_model, ((0,0), (0,0), (256, 256)), constant_values=0)
    return(brain_vol)


# CREATE MASKS FOR KSPACE OPERATIONS
def partial_fourier(kspsliderval1, kspsliderval2):
    global mask
    yvalue=int(kspsliderval1/100*512)
    mask= np.zeros([512,512], dtype=np.complex)
    mask[0:yvalue,0:512] = 1
    return mask 

def partial_echo(kspsliderval1, kspsliderval2):
    global mask
    xvalue=int(kspsliderval1/100*512)
    mask= np.zeros([512,512], dtype=np.complex)
    mask[0:512,0:xvalue] = 1
    return mask 

def central_kspace(kspsliderval1, kspsliderval2):
    global mask
    mask_radius = int(kspsliderval1/100*256)
    x,y = np.ogrid[:512,:512]
    mask= np.zeros([512,512], dtype=np.complex)
    mask_area_1 = (x-256)**2 + (y-256)**2 <=mask_radius*mask_radius
    mask[mask_area_1]=1
    return mask 

def peripheral_kspace(kspsliderval1, kspsliderval2):
    global mask
    mask_radius = int(kspsliderval1/100*256)
    x,y = np.ogrid[:512,:512]
    mask= np.ones([512,512], dtype=np.complex)
    mask_area_1 = (x-256)**2 + (y-256)**2 <=mask_radius*mask_radius
    mask[mask_area_1]=0
    return mask 


def zipper_artifact(kspsliderval1, kspsliderval2):
    global mask
    zipper_loc = int(kspsliderval1*512/100)
    mask= np.ones([512,512], dtype=np.complex)
    mask[:,zipper_loc] = 100
    return mask 

def nyquist_ghosts(kspsliderval1, kspsliderval2):
    global mask
    mask= np.ones([512,512], dtype=np.complex)
    for re in range (0,511,2):                    # 2 denotes alternate lines of KSpace are differently intensity modulated
        mask[re,:] = (kspsliderval1/100)        # kspasliderval1 Controls the intensity of the artifact  
    return mask 

def aliasing(kspsliderval1, kspsliderval2):
    global mask
    mask= np.zeros([512,512], dtype=np.complex)
    for tj in range(0,511,kspsliderval1):
        mask[:,tj] = 1
    return mask 


# CONSTRUCT THE IMAGE DATA USING ACQUISITION PARAMETERS AND K SPACE PARAMETERS

def create_brain_map(pseqcse, pseqfse, etlvalue, 
                     pseqmir, pseqpsir, pseqfir, iretlvalue, pseqspgr, 
                     fastmodenone, fastmodesense, sensefactor, fastmodecsense, csensefactor, 
                     acqplaneax, acqplanecor, acqplanesag, noofslices, slicethkness, slicegap, 
                     rloffset, apoffset, hfoffset, 
                     fovfreq, phasewrap, nophasewrap, acqfreq, acqphase, 
                     acqmodecart, acqmodeepi, epishots, acqmoderad, radialspokes, acqmodeprop, propblades, acqmodespir, spirinterleaves, 
                     faval, trval, teval, tival, bwval, nexval, filtsmooth, filtnormal, filtsharp, 
                     kspacefa, kspacepf, kspacepe, kspacec, kspacep, kspaceu, kspacez, 
                     iqmotion, iqspike, iqzipper, iqnyghosts, iqrfclip, iqdcoffset, iqrfshade, 
                     kspsliderval1, kspsliderval2, 
                     currslice, brightness, contrast):  
                     
    # DECLARE AND CLEAR ALL VARIABLES                     

    pseqcse=int(pseqcse); pseqfse=int(pseqfse); etlvalue=int(etlvalue); 
    pseqmir=int(pseqmir); pseqpsir=int(pseqpsir); pseqfir = int(pseqfir); iretlvalue = int(iretlvalue); pseqspgr=int(pseqspgr);
    fastmodenone=int(fastmodenone); fastmodesense=int(fastmodesense); sensefactor=int(sensefactor); fastmodecsense=int(fastmodecsense); csensefactor=int(csensefactor);
    acqplaneaxval=int(acqplaneax); acqplanecorval=int(acqplanecor); acqplanesagval=int(acqplanesag); noofslices=int(noofslices); slicethkness=int(slicethkness); slicegap=int(slicegap); 
    rloffset=int(rloffset); apoffset=int(apoffset); hfoffset=int(hfoffset);
    fovfreq=int(fovfreq); phasewrap=int(phasewrap); nophasewrap=int(nophasewrap); acqfreq=int(acqfreq); acqphase=int(acqphase);
    acqmodecart=int(acqmodecart); acqmodeepi=int(acqmodeepi); epishots=int(epishots); acqmoderad=int(acqmoderad); radialspokes=int(radialspokes); acqmodeprop=int(acqmodeprop); propblades=int(propblades); acqmodespir=int(acqmodespir); spirinterleaves=int(spirinterleaves);
    faval=int(faval); trval=int(trval); teval=int(teval); tival=int(tival); bwval=int(bwval); nexval=int(nexval); filtsmooth=int(filtsmooth); filtnormal=int(filtnormal); filtsharp=int(filtsharp);
    kspacefa=int(kspacefa); kspacepf=int(kspacepf); kspacepe=int(kspacepe); kspacec=int(kspacec); kspacep=int(kspacep); kspaceu=int(kspaceu); kspacez=int(kspacez);
    iqmotion=int(iqmotion); iqspike=int(iqspike); iqzipper=int(iqzipper); iqnyghosts=int(iqnyghosts); iqrfclip=int(iqrfclip); iqdcoffset=int(iqdcoffset); iqrfshade=int(iqrfshade); 
    kspsliderval1=int(kspsliderval1); kspsliderval2=int(kspsliderval2);
    currslice=int(currslice); brightness=int(brightness); contrast=int(contrast)

    # SUBROUTINE FOR AXIAL vs CORONAL vs SAGITTAL SLICES

    if acqplaneaxval==1:
        axial2axial(brain_model)
        move0=rloffset; move1=apoffset; move2=hfoffset; 

    if acqplanecorval==1:
        axial2coronal(brain_model)
        move0=rloffset; move1=hfoffset; move2=apoffset; 

    if acqplanesagval==1:
        axial2sagittal(brain_model)
        move0=apoffset; move1=hfoffset; move2=rloffset; 
    
    # IMAGE DATA RECONSTRUCTION
    global all_slices
    global slice_comp
    global image_data
    global mask_data
    global y
    global mask
    global traj_array
    global zoom_data
    global snr_value
    
    I0 = 512 + move0
    I1 = 512 + move1
    I2 = 512 + move2
    fovfreq = int(fovfreq*2)
    fovphase = fovfreq 

    acqfreq=int(acqfreq)
    acqphase=int(acqphase)
    slice_data_cropped=np.empty((fovphase,fovfreq),dtype=np.uint8)
    slice_data_upsampled=np.empty((512,512),dtype=np.uint8)
    kspace_displ = np.zeros_like([512,512], dtype=np.complex64)
    traj_array=np.zeros((512,512,3),dtype=np.uint8)
    traj_array_text=np.zeros((512,512,3),dtype=np.uint8)

    freqstart = int(I0-int(fovfreq/2))
    freqend = int(freqstart+fovfreq)
    phasestart = int(I1-int(fovphase/2))
    phaseend = int(phasestart+fovphase)


    
    # MASK FOR KSPACE OPERATIONS IS CREATED HERE
    if kspacepf==1:
        mask=partial_fourier(kspsliderval1, kspsliderval2)
        
    if kspacepe==1:
        mask=partial_echo(kspsliderval1, kspsliderval2)
        
    if kspacec==1:
        mask=central_kspace(kspsliderval1, kspsliderval2)
        
    if kspacep==1:
        mask=peripheral_kspace(kspsliderval1, kspsliderval2)
                
    if iqzipper==1:
        mask=zipper_artifact(kspsliderval1, kspsliderval2)

    if iqnyghosts==1:
        mask=nyquist_ghosts(kspsliderval1, kspsliderval2)


# K SPACE TRAJECTORIES ARE CALCULATED HERE
    
    # K SPACE TRAJECTORY FOR CARTESIAN CONVENTIONAL SPIN ECHO
    if (pseqcse == 1 and acqmodecart == 1) or (pseqmir == 1 and acqmodecart == 1) or (pseqpsir == 1 and acqmodecart == 1) or (pseqspgr == 1 and acqmodecart == 1) :
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)
        for nnn in range (4, 255, 4):
            for nno in range (15, 500, 5):
                cv2.line(traj_array, (nno,nnn*2), (nno+2,nnn*2), (155,155,155), 1)


        cv2.putText(traj_array_text, '<< '+str(acqphase)+' Phase Encodes >>', (140,465), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv2.putText(traj_array_text, 'Each Phase Encode in one TR', (130,480), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        RotM = cv2.getRotationMatrix2D((256,256), 90, 1)
        traj_array_text = cv2.warpAffine(traj_array_text, RotM, (512,512))
        cv2.putText(traj_array_text, '<< '+str(acqfreq)+' Frequency Encodes >>', (130,477), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)

        cv2.rectangle(traj_array, (125,465), (390,475), (0,0,0),-1)
        cv2.rectangle(traj_array, (450,125), (485,390), (0,0,0),-1)

        traj_array = traj_array_text + traj_array

    # K SPACE TRAJECTORY FOR CARTESIAN FAST SPIN ECHO OR FAST INVERSION RECOVERY
    if pseqfse == 1 or pseqfir==1 :
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)
        Cbar = { 1: [142, 64, 171, 1, 89, 178, 74, 62, 220, 251, 67, 210, 239, 9, 182, 130, 231, 93, 199, 133, 226, 1, 36, 156, 52, 10, 78, 12, 166, 141, 159, 72, 155, 229, 71, 227, 182, 165, 233, 97, 135, 0, 59, 240, 182, 48, 20, 46, 16, 208, 107, 75],
                 2: [78, 12, 166, 141, 159, 72, 155, 229, 71, 227, 182, 165, 233, 97, 135, 0, 59, 240, 182, 48, 20, 46, 16, 208, 107, 75, 98, 1, 136, 238, 121, 201, 42, 248, 79, 69, 228, 47, 121, 109, 132, 129, 214, 58, 39, 149, 155, 3, 109, 124, 93, 8],
                 3: [98, 1, 136, 238, 121, 201, 42, 248, 79, 69, 228, 47, 121, 109, 132, 129, 214, 58, 39, 149, 155, 3, 109, 124, 93, 8, 142, 64, 171, 1, 89, 178, 74, 62, 220, 251, 67, 210, 239, 9, 182, 130, 231, 93, 199, 133, 226, 1, 36, 156, 52, 10]}
        lin=0
        if pseqfir==1:
            etlvalue = iretlvalue

        etljump = 60/(etlvalue)

        for nno in range (8, 504, 8):
            if lin < etljump:
                for nnp in range (15, 500, 5):
                    cv2.line(traj_array, (nnp,nno), (nnp+2,nno), (Cbar[1][lin],Cbar[2][lin],Cbar[3][lin]), 1)
                lin = lin+1
            else:
                lin = 0
                for nnp in range (15, 500, 5):
                    cv2.line(traj_array, (nnp,nno), (nnp+2,nno), (Cbar[1][lin],Cbar[2][lin],Cbar[3][lin]), 1)
                lin = lin+1
       
        cv2.putText(traj_array_text, '<< '+str(acqphase)+' Phase Encodes >>', (140,465), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv2.putText(traj_array_text, str(int(etlvalue))+' Phase Encodes in one TR', (130,480), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        RotM = cv2.getRotationMatrix2D((256,256), 90, 1)
        traj_array_text = cv2.warpAffine(traj_array_text, RotM, (512,512))
        cv2.putText(traj_array_text, '<< '+str(acqfreq)+' Frequency Encodes >>', (130,477), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)

        cv2.rectangle(traj_array, (125,465), (390,475), (0,0,0),-1)
        cv2.rectangle(traj_array, (450,125), (485,390), (0,0,0),-1)

        traj_array = traj_array_text + traj_array

    # K SPACE TRAJECTORY FOR ECHO PLANAR IMAGING
    if acqmodeepi == 1:
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)
        
        if epishots==1:
            for nnn in range (4, 128, 4):
                for nno in range (15, 500, 5):
                    cv2.line(traj_array, (nno,nnn*4), (nno+2,nnn*4), (10, 198, 245), 1)

            for nnn in range (4, 120, 8):
                cv2.line(traj_array, (15,nnn*4+16), (15,nnn*4+32), (10, 198, 245), 1)
                cv2.line(traj_array, (500,nnn*4), (500,nnn*4+16), (10, 198, 245), 1)

            cv2.putText(traj_array_text, '<< '+str(acqphase)+' Phase Encodes >>', (140,458), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
            cv2.putText(traj_array_text, 'All Phase Encodes in one TR', (130,473), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
            RotM = cv2.getRotationMatrix2D((256,256), 90, 1)
            traj_array_text = cv2.warpAffine(traj_array_text, RotM, (512,512))
            cv2.putText(traj_array_text, '<< '+str(acqfreq)+' Frequency Encodes >>', (130,477), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)

            cv2.rectangle(traj_array, (125,465), (390,475), (0,0,0),-1)
            cv2.rectangle(traj_array, (445,125), (480,390), (0,0,0),-1)

        if epishots==2:
            for nnn in range (4, 128, 8):
                for nno in range (10, 490, 5):
                    cv2.line(traj_array, (nno,nnn*4), (nno+2,nnn*4), (10, 198, 245), 1)
                for nno in range (20, 500, 5):
                    cv2.line(traj_array, (nno,nnn*4+16), (nno+2,nnn*4+16), (186, 184, 43), 1)


            for nnn in range (4, 116, 16):
                cv2.line(traj_array, (10,nnn*4+32), (10,nnn*4+64), (10, 198, 245), 1)
                cv2.line(traj_array, (490,nnn*4), (490,nnn*4+32), (10, 198, 245), 1)
                cv2.line(traj_array, (500,nnn*4+16), (500,nnn*4+48), (186,184,1), 1)
                cv2.line(traj_array, (20,nnn*4+48), (20,nnn*4+80), (186,184,1), 1)

            cv2.putText(traj_array_text, '<< '+str(acqphase)+' Phase Encodes >>', (140,458), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
            cv2.putText(traj_array_text, '     Acquired in two TRs', (130,473), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
            RotM = cv2.getRotationMatrix2D((256,256), 90, 1)
            traj_array_text = cv2.warpAffine(traj_array_text, RotM, (512,512))
            cv2.putText(traj_array_text, '<< '+str(acqfreq)+' Frequency Encodes >>', (130,477), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
            traj_array = traj_array_text + traj_array

            cv2.rectangle(traj_array, (125,465), (390,475), (0,0,0),-1)
            cv2.rectangle(traj_array, (445,125), (480,390), (0,0,0),-1)
            
        traj_array = traj_array_text + traj_array

    # K SPACE TRAJECTORY FOR RADIAL IMAGING
    if acqmoderad == 1:
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)

        for i in range (0, 360, radialspokes):
            for j in range (250):
                shift_x=int(np.sin(np.radians(i))*j)
                shift_y=int(np.cos(np.radians(i))*j)
                cv2.line(traj_array, (255,255), ((255+shift_x),(255+shift_y)), (255,255,255), 1)                                  # RF axis line

        cv2.putText(traj_array_text, 'Radial spokes every '+str(int(radialspokes))+ ' deg', (250,495), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv2.rectangle(traj_array, (240,480), (490,500), (0,0,0),-1)
        traj_array = traj_array_text + traj_array

    # K SPACE TRAJECTORY FOR PROPELLER IMAGING

    if acqmodeprop == 1:
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)
        traj_array_frame = np.zeros((512,512,3),dtype=np.uint8)
        bladeangle=int(360/propblades)
        bladelines = 8
        blademoves = int(180/bladeangle)
        bladestart = 256-4*bladelines
        bladeend = 256+4*bladelines

        for tw in range (bladestart, bladeend, 4):
            cv2.line(traj_array_frame, (15,tw), (497,tw), (255,255,255), 1)                                  # RF axis line

        for tx in range (1, (blademoves+1)):
            RotM = cv2.getRotationMatrix2D((256,256), (tx*bladeangle), 1)
            traj_array_frame_curr = cv2.warpAffine(traj_array_frame, RotM, (512,512))
            traj_array = traj_array + traj_array_frame_curr
            
        cv2.putText(traj_array_text, '16 lines acquired every '+str(int(bladeangle))+ ' deg', (250,495), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv2.rectangle(traj_array, (240,480), (490,510), (0,0,0),-1)
        traj_array = traj_array_text + traj_array


    # K SPACE TRAJECTORY FOR SPIRAL IMAGING
    if acqmodespir == 1:
        r=0
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)

        if spirinterleaves == 1:
            for ty in range (0,7800):
                r = r+0.03
                x = int(r* np.cos(np.deg2rad(ty)))
                y = int(r* np.sin(np.deg2rad(ty)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=3)

        if spirinterleaves == 2:
            for ty in range (0,4500):
                r = r+0.05
                x = int(r* np.cos(np.deg2rad(ty)))
                y = int(r* np.sin(np.deg2rad(ty)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=3)
                x = int(r* np.cos(np.deg2rad(ty+180)))
                y = int(r* np.sin(np.deg2rad(ty+180)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=3)

        if spirinterleaves == 3:
            for ty in range (0,4500):
                r = r+0.05
                x = int(r* np.cos(np.deg2rad(ty)))
                y = int(r* np.sin(np.deg2rad(ty)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)
                x = int(r* np.cos(np.deg2rad(ty+120)))
                y = int(r* np.sin(np.deg2rad(ty+120)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)
                x = int(r* np.cos(np.deg2rad(ty+240)))
                y = int(r* np.sin(np.deg2rad(ty+240)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)

        if spirinterleaves == 4:
            for ty in range (0,3900):
                r = r+0.06
                x = int(r* np.cos(np.deg2rad(ty)))
                y = int(r* np.sin(np.deg2rad(ty)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)
                x = int(r* np.cos(np.deg2rad(ty+90)))
                y = int(r* np.sin(np.deg2rad(ty+90)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)
                x = int(r* np.cos(np.deg2rad(ty+180)))
                y = int(r* np.sin(np.deg2rad(ty+180)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)
                x = int(r* np.cos(np.deg2rad(ty+270)))
                y = int(r* np.sin(np.deg2rad(ty+270)))
                cv2.circle(traj_array, (x+256,y+256), radius=0, color=(180, 180, 180), thickness=2)

        cv2.putText(traj_array_text, str(int(spirinterleaves))+ ' spirals acquired', (340,495), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv2.rectangle(traj_array, (330,480), (490,510), (0,0,0),-1)
        traj_array = traj_array_text + traj_array

    # K SPACE TRAJECTORY FOR PARALLEL IMAGING
    if fastmodesense == 1 and acqmodecart==1:
        traj_array = np.zeros((512,512,3),dtype=np.uint8)
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)
        for nnn in range (4, 252, (4*sensefactor)):
            for nno in range (15, 496, 5):
                cv2.line(traj_array, (nno,nnn*2), (nno+2,nnn*2), (155,155,155), 1)

        cv2.putText(traj_array_text, '<< '+str(int(acqphase/sensefactor))+' Phase Encodes >>', (140,464), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        cv2.putText(traj_array_text, 'using multiple coil elements', (130,480), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        RotM = cv2.getRotationMatrix2D((256,256), 90, 1)
        traj_array_text = cv2.warpAffine(traj_array_text, RotM, (512,512))
        cv2.putText(traj_array_text, '<< '+str(acqfreq)+' Frequency Encodes >>', (130,477), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)

        cv2.rectangle(traj_array, (125,465), (390,475), (0,0,0),-1)
        cv2.rectangle(traj_array, (450,125), (485,390), (0,0,0),-1)

        traj_array = traj_array_text + traj_array

    # K SPACE TRAJECTORY FOR COMPRESSED SENSE IMAGING
    if fastmodecsense == 1 and acqmodecart==1:
        traj_array = np.zeros((512,512,3),dtype=np.uint8)
        traj_array_text = np.zeros((512,512,3),dtype=np.uint8)
        sample_qtty = csensefactor/100*512*512

        for i in range (int(sample_qtty)):
            sample_x=np.random.randint(0, 511)    
            sample_y=np.random.randint(0, 511)    
            traj_array[sample_y][sample_x] = 255 

        cv2.rectangle(traj_array, (125,460), (400,485), (0,0,0),-1)
        cv2.putText(traj_array_text, '  '+str(csensefactor*10)+' % of kSpace is Acquired ', (130,477), cv2.FONT_HERSHEY_PLAIN, 1, 255, 1)
        traj_array = traj_array_text + traj_array

    

    # SIGNAL INTENSITY VALUES ARE CALCULATED HERE
    def signal_calc(T1, T2, T2star, PD):
        FA = faval; TR = trval; TE = teval; TI = tival
        if FA ==0:
            FA=1
        if TR ==0:
            TR=1
        if TE ==0:
            TE=1
        if TI ==0:
            TI=1
            
        if pseqcse==1 or pseqfse ==1:
            signal_data = PD * (1 - np.exp(-TR/T1)) * np.exp(-TE/T2)
            signal_data = np.abs(signal_data)
            return signal_data
        
        if pseqmir==1:
            signal_data = PD * (1 - 2*np.exp(-TI/T1) * np.exp(-TR/T1)) 
            signal_data_TI = np.abs(signal_data)
            signal_final = signal_data_TI * np.exp(-TE/T2)
            return signal_final

        if pseqpsir==1:
            signal_final = PD * (1 - 2*np.exp(-TI/T1) * np.exp(-TR/T1)) * np.exp(-TE/T2)
            return signal_final

        if pseqspgr==1: 
            signal_data = PD * np.exp(-TE/T2star) * np.sin(np.deg2rad(FA)) * (1 - np.exp(-TR/T1)) / (1 - np.cos(np.deg2rad(FA)) * np.exp(-TR/T1))
            return signal_data
             
    # SIGNAL IS CALCULATED FOR INDIVIDUAL TISSUES HERE
    for tiss in range(1,6):
        T1 = parameters[tiss][2]
        T2 = parameters[tiss][3]
        T2star = parameters[tiss][4]
        PD = parameters[tiss][5]
        
        parameters[tiss][9]=int(256*signal_calc(T1, T2, T2star, PD))


    signal=np.zeros((512,512),dtype=np.uint8)
    slice_data_padded=np.zeros((1024,1024),dtype=np.uint8)

    # SLICE DATA IS EXTRACTED FROM THE VOLUME HERE
    y= I2 - int(noofslices*(slicethkness+slicegap))  # Not multiplying by 2 as each mm is 2 slices

    for i in range(0,int(noofslices)):
        slice_data=np.empty((512,512, int(slicethkness*2)),dtype=np.uint8)
        slice_data_final=np.empty((512,512),dtype=np.uint8)
        for j in range(0,int(slicethkness*2)):
            loc=int(y+j)
            slice_comp[:,:]=brain_vol[:,:,loc]
                        
            # SIGNAL INTENSITY VALUES ARE ASSIGNED TO DIFFERENT TISSUES HERE
            for tissue in range(1,6):
                signal[slice_comp ==tissue] = parameters[tissue][9]

            if pseqpsir ==1:
                signal = (signal*1.28)+128

            slice_data[:,:,j] = signal[:,:]

        # AVERAGE THE SLICE_COMP TO GET SLICE_DATA 
        slice_data_final= np.mean(slice_data, axis=2)

        # ZERO PADDING AND FOV CROPPING IS DONE HERE             
        slice_data_padded[:,:]= np.pad(slice_data_final, ((256, 256), (256, 256)), constant_values=0)

        if phasewrap ==1:
            image_left = np.uint8((slice_data_padded[:, 0:phasestart])*0.3)
            image_right = np.uint8((slice_data_padded[:, phaseend:1023])*0.3)
            left_im_v,left_im_h  = image_left.shape
            right_im_v,right_im_h  = image_right.shape
            slice_data_padded[:,(phaseend-left_im_h):phaseend]=slice_data_padded[:,(phaseend-left_im_h):phaseend]+image_left
            slice_data_padded[:,phasestart:(phasestart+right_im_h)]=slice_data_padded[:,phasestart:(phasestart+right_im_h)]+image_right

        slice_data_cropped[:,:] = slice_data_padded[phasestart:phaseend, freqstart:freqend]
        slice_data_upsampled[:,:] = cv2.resize(slice_data_cropped, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

        # SOME KSPACE OPERATIONS THAT ARE DONE DIRECTLY ON SLICE_DATA

        if (kspacefa==1):          
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            gibbs_mask = np.ones_like(shifted_kspace)
            gibbs_fact_phase = int(acqphase/2)
            gibbs_fact_freq = int(acqfreq/2)
            gibbs_mask[0:(256-gibbs_fact_freq),0:511] = 0
            gibbs_mask[(256+gibbs_fact_freq):511,0:511] = 0
            gibbs_mask[0:511,0:(256-gibbs_fact_phase)] = 0
            gibbs_mask[0:511,(256+gibbs_fact_phase):511] = 0

            gibbsed_kspace = shifted_kspace * gibbs_mask
            recon_image= reconstruct(gibbsed_kspace)
            mask_displ=np.uint8(kspace_displ)

        if (kspacepf==1 or kspacepe==1 or kspacec==1 or kspacep==1 or iqzipper==1 or iqnyghosts==1):          
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            processed_kspace = shifted_kspace * mask
            recon_image = reconstruct(processed_kspace)
            mask_displ=np.uint8(kspace_displ*mask)

        if kspaceu==1:
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            new_kspace = np.empty((int(512/kspsliderval1), 512), dtype=np.float32)
            new_displ = np.empty((int(512/kspsliderval1), 512), dtype=np.float32)
            mask_displ = np.zeros_like(shifted_kspace)

            for rd in range (0,511):
                if (rd % kspsliderval1 == 0):
                    lin_no = (int(rd/kspsliderval1)-1)
                    new_kspace[lin_no,:] = shifted_kspace[rd,:]
                    new_displ[lin_no,:] = kspace_displ[rd,:]
            undersampled_image = reconstruct(new_kspace)
            rows_undersampled, cols_undersampled= undersampled_image.shape
            im_start = int(256-(rows_undersampled/2))
            im_end = int(im_start+rows_undersampled)
            recon_image = np.zeros((512,512), dtype=np.float)
            recon_image[im_start:im_end, :] = undersampled_image[:,:]
            mask_displ[im_start:im_end, :] = new_displ[:,:]

        if kspacez==1:
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            mask = np.zeros_like(shifted_kspace)
            for re in range (0,511,kspsliderval1):                    # 2 denotes alternate lines of KSpace are differently intensity modulated
                mask[re,:] = 1   
            new_kspace = shifted_kspace * mask
            recon_image= reconstruct(new_kspace)
            mask_displ=kspace_displ * mask

        if iqmotion==1:
            fsefactor=4             # DECIDES No OF TRs - ASSUMING FSE FACTOR IS 4 (4 LINES OF KSPACE FILLED IN ONE TR)
            fse_traversals=int(512/fsefactor)
            if (512%fsefactor !=1):
                fse_traversals = fse_traversals+1    
            jump_point = int(fse_traversals*kspsliderval2/100)  # KSPSLIDERVAL2 DECIDES WHERE IN THE KSPACE THE MOVEMENT OCCURS (IN % OF TRs)
            moved_image_data = np.roll(slice_data_final, kspsliderval1, axis=1)  # KSPSLIDERVAL1 REPRESENTS AMOUNT OF MOVEMENT IN X AXIS 
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            kspace_moved, moved_kspace = gen_k_space(moved_image_data)            
            new_kspace = shifted_kspace
            rr=0; rp=jump_point
            while rp<= fse_traversals:
                while rr < 512:
                    new_kspace[:,rr] = moved_kspace[:,rr]
                    rr=rr+4                                         # 4 is t
                rp = rp+1
            recon_image= reconstruct(new_kspace) 
            mask_displ=kspace_displ
            
        if iqspike==1:
            spike_loc_x = int(512*kspsliderval1/100)
            spike_loc_y = int(512*kspsliderval2/100)
            mask= np.zeros([512,512], dtype=np.complex)
            mask[spike_loc_y, spike_loc_x] = 12237710+0j    
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            new_kspace = shifted_kspace + mask
            recon_image= reconstruct(new_kspace) 
            mask_displ=mask

        if iqrfclip==1:
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            shifted_kspace[shifted_kspace>(kspsliderval1/100* np.max(shifted_kspace))]=0
            shifted_kspace[shifted_kspace<(kspsliderval1/100* np.min(shifted_kspace))]=0
            new_kspace = shifted_kspace
            recon_image= reconstruct(new_kspace) 
            mask_displ=kspace_displ
                        
        if iqdcoffset==1:
            image_source = slice_data_final
            image_source[254:258,254:258]=255
            kspace_displ, shifted_kspace = gen_k_space(slice_data_upsampled)
            recon_image=image_source
            mask_displ=kspace_displ

        if iqrfshade==1:
            shade_qtty=int(kspsliderval1*256/100)
            mask =np.uint8(np.tile(np.linspace(0, shade_qtty, 512), (512, 1)))
            RotMatrx = cv2.getRotationMatrix2D((256,256), (kspsliderval2*90), 1)
            mask = cv2.warpAffine(mask, RotMatrx, (512,512))
            image_source= np.asarray(slice_data_upsampled,dtype='float')
            recon_image = image_source * mask/256
            mask_displ, shifted_kspace = gen_k_space(recon_image)

        image_data [:,:,i] = recon_image
        mask_data [:,:,i] = mask_displ

        y=y+int(slicethkness*2)+int(slicegap*2)

        all_slices= image_data

# PULSE SEQUENCE DIAGRAMS ARE GENERATED HERE

def psd_calc_se(TR, TE, ETL, pseqfse):
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1
    
    Xlim=160; rfy  = 125; addly=190; rf90x = 140; rf180x = int(rf90x+(Xlim/2)); rf902x = 480; sly = 250; pex = 20; pey = 310; fey = 370; signy = 430; fidx=rf90x+9; echox = rf90x + Xlim
    cv2.putText(psd_array,'SPIN ECHO', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(psd_array, (60,rfy), (450,rfy), (255,255,255), 1)                                  # RF axis line
    cv2.line(psd_array, (60,sly), (450,sly), (255,255,255), 1)                                  # SL axis line
    cv2.line(psd_array, (60,addly), (450,addly), (255,255,255), 1)                              # Prep axis line
    cv2.line(psd_array, (60,pey), (450,pey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (60,fey), (450,fey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (60,signy), (450,signy), (255,255,255), 1)                              # Signal axis line

    cv2.line(psd_array, (460,rfy), (500,rfy), (255,255,255), 1)                                  # RF axis line
    cv2.line(psd_array, (460,sly), (500,sly), (255,255,255), 1)                                  # SL axis line
    cv2.line(psd_array, (460,addly), (500,addly), (255,255,255), 1)                              # Addl axis line
    cv2.line(psd_array, (460,pey), (500,pey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (460,fey), (500,fey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (460,signy), (500,signy), (255,255,255), 1)                              # Signal axis line

    cv2.line(psd_array, ((rf90x+8),50), ((rf90x+8),470), (55,55,55), 1)                         # Vert Line RF
    cv2.line(psd_array, (rf180x,50), (rf180x,450), (55,55,55), 1)                               # Vert Line 180
    cv2.line(psd_array, (echox,50), (echox,450), (55,55,55), 1)                                 # Vert Line Echo
    cv2.line(psd_array, ((rf902x-8),50), ((rf902x-8),470), (55,55,55), 1)                       # Vert Line RF2
    cv2.rectangle(psd_array,((rf90x-8), (rfy+1)), ((rf90x+8), (rfy-30)), (250, 145, 40), -1)        # RF90
    cv2.rectangle(psd_array,((rf180x-8), (rfy+1)), ((rf180x+8), (rfy-50)), (250, 145, 40), -1)      # RF180
    cv2.rectangle(psd_array,((rf902x-8), (rfy+1)), ((rf902x+8), (rfy-30)), (250, 145, 40), -1)      # RF90


    cv2.rectangle(psd_array,((rf90x-8), (sly-1)), ((rf90x+8), (sly-10)), (82, 82, 82), -1)          # Slice 1
    cv2.rectangle(psd_array,((rf180x-8), (sly-1)), ((rf180x+8), (sly-10)), (82, 82, 82), -1)        # Slice 2
    cv2.rectangle(psd_array,((rf902x-8), (sly-1)), ((rf902x+8), (sly-10)), (82, 82, 82), -1)        # Slice 3

    for pegrdraw in range (1,6):
        cv2.rectangle(psd_array,(fidx+pex-10, (pey-3*pegrdraw)), (fidx+pex+10, (pey+3*pegrdraw)), (191, 189, 191), 1)  # PE Gradient Steps


    cv2.putText(psd_array,'RF', (25, rfy), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Addl', (20, addly), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Slice', (20, sly), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'PE', (25, pey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'FE', (25, fey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Signal', (15, signy), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.putText(psd_array, '90', (rf90x-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)
    cv2.putText(psd_array, '180', (rf180x-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)
    cv2.putText(psd_array, '90', (rf902x-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)

    cv2.rectangle(psd_array,((echox-12), (fey-1)), ((echox+12), (fey-10)), (28, 125, 173), -1)      # FE Gradient

    cv2.putText(psd_array,'TE/2', (rf180x-20, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'TE = '+str(int(TE))+' ms', (echox-20, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'TR = '+str(int(TR))+' ms', ((rf902x-100), 490), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)


    fid_contour= np.array([[fidx, (signy+25)], [fidx,(signy-25)], [(fidx+15), signy]], np.int32)                # FID Signal
    cv2.polylines(psd_array, [fid_contour], True, (255,0,0), thickness=1)   
    echo_contour=np.array([[(echox-10),signy], [echox,(signy+25)], [(echox+10),signy ], [echox,(signy-25)]])    # Echo Signal
    cv2.polylines(psd_array, [echo_contour], True, (255,0,0), thickness=1)   


def psd_calc_ir(TR, TE, TI, ETL):
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1
    if TI ==0:
        TI=1

    Xlim=160; rfy  = 125; rf90x = 175; rf180x = int(rf90x+(Xlim/2)); rf902x = 480; rfinvx = 80; sly = 190; pex = 30; pey = 250; fey = 310; signy = 370; fidx=rf90x+9; echox = rf90x + Xlim

    cv2.putText(psd_array,'INVERSION RECOVERY', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(psd_array, (60,rfy), (460,rfy), (255,255,255), 1)                                  # RF axis line
    cv2.line(psd_array, (60,sly), (460,sly), (255,255,255), 1)                                  # SL axis line
    cv2.line(psd_array, (60,pey), (460,pey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (60,fey), (460,fey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (60,signy), (460,signy), (255,255,255), 1)                              # Signal axis line

    cv2.line(psd_array, (460,rfy), (500,rfy), (255,255,255), 1)                                  # RF axis line
    cv2.line(psd_array, (460,sly), (500,sly), (255,255,255), 1)                                  # SL axis line
    cv2.line(psd_array, (460,pey), (500,pey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (460,fey), (500,fey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (460,signy), (500,signy), (255,255,255), 1)                              # Signal axis line

    cv2.line(psd_array, ((rfinvx+8),50), ((rfinvx+8),470), (55,55,55), 1)                         # Vert Line RF
    cv2.line(psd_array, ((rf90x-8),50), ((rf90x-8),470), (55,55,55), 1)                         # Vert Line RF
    cv2.line(psd_array, ((rf90x+8),50), ((rf90x+8),470), (55,55,55), 1)                         # Vert Line RF
    cv2.line(psd_array, (rf180x,50), (rf180x,400), (55,55,55), 1)                               # Vert Line 180
    cv2.line(psd_array, (echox,50), (echox,400), (55,55,55), 1)                                 # Vert Line Echo
    cv2.line(psd_array, ((rf902x-8),50), ((rf902x-8),450), (55,55,55), 1)                       # Vert Line RF2

    cv2.rectangle(psd_array,((rfinvx-8), (rfy+1)), ((rfinvx+8), (rfy-50)), (250, 145, 40), -1)
    cv2.rectangle(psd_array,((rf90x-8), (rfy+1)), ((rf90x+8), (rfy-30)), (250, 145, 40), -1)
    cv2.rectangle(psd_array,((rf180x-8), (rfy+1)), ((rf180x+8), (rfy-50)), (250, 145, 40), -1)
    cv2.rectangle(psd_array,((rf902x-8), (rfy+1)), ((rf902x+8), (rfy-50)), (250, 145, 40), -1)

    cv2.rectangle(psd_array,((rfinvx-8), (sly+1)), ((rfinvx+8), (sly-10)), (82, 82, 82), -1)
    cv2.rectangle(psd_array,((rf90x-8), (sly-1)), ((rf90x+8), (sly-10)), (82, 82, 82), -1)
    cv2.rectangle(psd_array,((rf180x-8), (sly-1)), ((rf180x+8), (sly-10)), (82, 82, 82), -1)
    cv2.rectangle(psd_array,((rf902x-8), (sly-1)), ((rf902x+8), (sly-10)), (82, 82, 82), -1)

    for pegrdraw in range (1,6):
        cv2.rectangle(psd_array,(fidx+pex-10, (pey-3*pegrdraw)), (fidx+pex+10, (pey+3*pegrdraw)), (191, 189, 191), 1)  # PE Gradient Steps

    cv2.putText(psd_array,'RF', (25, rfy), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Slice', (20, sly), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'PE', (25, pey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'FE', (25, fey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Signal', (15, signy), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.putText(psd_array, '180', (rfinvx-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)
    cv2.putText(psd_array, '90', (rf90x-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)
    cv2.putText(psd_array, '180', (rf180x-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)
    cv2.putText(psd_array, '180', (rf902x-10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)


    cv2.rectangle(psd_array,((echox-12), (fey-1)), ((echox+12), (fey-10)), (28, 125, 173), -1)

    cv2.putText(psd_array,'TE/2', (rf180x-20, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'TE = '+str(int(TE))+' ms', (echox-20, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'TR = '+str(int(TR))+' ms', ((rf902x-100), 470), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)


    fid_contour= np.array([[fidx, (signy+25)], [fidx,(signy-25)], [(fidx+15), signy]], np.int32)
    cv2.polylines(psd_array, [fid_contour], True, (255,0,0), thickness=1)   
    echo_contour=np.array([[(echox-10),signy], [echox,(signy+25)], [(echox+10),signy ], [echox,(signy-25)]])
    cv2.polylines(psd_array, [echo_contour], True, (255,0,0), thickness=1)   



def psd_calc_ge(FA, TR, TE):
    if FA ==0:
        FA=1
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1

    Xlim=60; rfy  = 125; rf90x = 75; rf902x = 480; sly = 190; pex = 20; pey = 250; fey = 310; signy = 370; fidx=rf90x+9; echox = rf90x + Xlim

    cv2.putText(psd_array,'GRADIENT ECHO', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(psd_array, (60,rfy), (500,rfy), (255,255,255), 1)                                  # RF axis line
    cv2.line(psd_array, (60,sly), (500,sly), (255,255,255), 1)                                  # SL axis line
    cv2.line(psd_array, (60,pey), (500,pey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (60,fey), (500,fey), (255,255,255), 1)                                  # PE axis line
    cv2.line(psd_array, (60,signy), (500,signy), (255,255,255), 1)                              # Signal axis line


    cv2.line(psd_array, ((rf90x-8),50), ((rf90x-8),470), (55,55,55), 1)                         # Vert Line RF
    cv2.line(psd_array, ((rf90x+8),50), ((rf90x+8),470), (55,55,55), 1)                         # Vert Line RF
    cv2.line(psd_array, (echox,50), (echox,400), (55,55,55), 1)                                 # Vert Line Echo
    cv2.line(psd_array, ((rf902x-8),50), ((rf902x-8),450), (55,55,55), 1)                       # Vert Line RF2

    cv2.rectangle(psd_array,((rf90x-8), (rfy+1)), ((rf90x+8), (rfy-30)), (250, 145, 40), -1)
    cv2.rectangle(psd_array,((rf902x-8), (rfy+1)), ((rf902x+8), (rfy-30)), (250, 145, 40), -1)

    cv2.rectangle(psd_array,((rf90x-8), (sly-1)), ((rf90x+8), (sly-10)), (82, 82, 82), -1)
    cv2.rectangle(psd_array,((rf902x-8), (sly-1)), ((rf902x+8), (sly-10)), (82, 82, 82), -1)

    for pegrdraw in range (1,6):
        cv2.rectangle(psd_array,(fidx+pex-10, (pey-3*pegrdraw)), (fidx+pex+10, (pey+3*pegrdraw)), (191, 189, 191), 1)  # PE Gradient Steps

    cv2.putText(psd_array,'RF', (25, rfy), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Slice', (20, sly), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'PE', (25, pey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'FE', (25, fey), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'Signal', (15, signy), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.rectangle(psd_array,((fidx+pex-10), (fey+1)), ((fidx+pex+10), (fey+10)), (28, 125, 173), -1)
    cv2.rectangle(psd_array,((fidx+pex+10), (fey-1)), ((fidx+pex+50), (fey-10)), (28, 125, 173), -1)
    cv2.putText(psd_array,'FA = '+str(int(FA)), (rf90x-10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (250, 145, 40), 1)
    cv2.putText(psd_array,'TE = '+str(int(TE))+' ms', (echox-20, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(psd_array,'TR = '+str(int(TR))+' ms', ((rf902x-100), 470), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    fid_contour= np.array([[(fidx), (signy+25)], [(fidx),(signy-25)], [(fidx+pex+10), signy]], np.int32)
    cv2.polylines(psd_array, [fid_contour], True, (255,0,0), thickness=1)   
    echo_contour=np.array([[(fidx+pex+10), signy], [echox,(signy+15)], [(fidx+pex+50),signy ], [echox,(signy-15)]])
    cv2.polylines(psd_array, [echo_contour], True, (255,0,0), thickness=1)   


# TISSUE RELAXATION CURVES ARE GENERATED HERE

def relax_calc_se(TR, TE):
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1

    XlimitTR = int(TR*1.1)                                                                                  # 10% more than the TR
    if XlimitTR < 3000:
        XlimitTR = 3000
    XscaleTR = XlimitTR/400
    scale_factor_t1=1.6

    if TE<100:
        XlimitTE = 100
    else:
        XlimitTE = TE*1.1                                                                                  # 10% more than the TE

    XscaleTE = XlimitTE/400

    cv2.line(relax_array, (1,236), (512,236), (55,55,55), 1)                                                            # Separator line
    cv2.putText(relax_array,'T1 RELAXATION', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)                    # T1 Title    
    cv2.line(relax_array, (20,210), (470,210), (255,255,255), 1)                                                        # X axis line
    cv2.line(relax_array, (20,30), (20,210), (255,255,255), 1)                                                          # Y axis line
    cv2.line(relax_array, ((20+int(TR/XscaleTR)), 210), ((20+int(TR/XscaleTR)), 40), (255,255,255), 1)                  # TR line
    cv2.putText(relax_array, str(int(TR))+'ms', (int(TR/XscaleTR), 230), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)   # TR Value
    cv2.putText(relax_array, 'TR', (int(TR/XscaleTR)+24, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)              # TR Label

    cv2.putText(relax_array,'T2 RELAXATION', (200, 265), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)                   # T2 Title
    cv2.line(relax_array, (20,460), (470,460), (255,255,255), 1)                                                        # RF axis line
    cv2.line(relax_array, (20,260), (20,460), (255,255,255), 1)                                                         # RF axis line
    cv2.line(relax_array, ((20+int(TE/XscaleTE)), 460), ((20+int(TE/XscaleTE)),272), (255,255,255), 1)                  # TE line
    cv2.putText(relax_array, str(int(TE))+'ms', (int(TE*XscaleTE), 480), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)   # TE Value
    cv2.putText(relax_array, 'TE', (20+int(TE/XscaleTE)+5, 290), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)             # TE Label

    cv2.putText(relax_array, 'CSF', (100, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[2][6], parameters[2][7], parameters[2][8]), 1)
    cv2.putText(relax_array, 'Grey Matter', (150, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[3][6], parameters[3][7], parameters[3][8]), 1)
    cv2.putText(relax_array, 'White Matter', (280, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[4][6], parameters[4][7], parameters[4][8]), 1)
    cv2.putText(relax_array, 'Fat', (420, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[5][6], parameters[5][7], parameters[5][8]), 1)

    for tt in range (2,6):
        T1=parameters[tt][2]
        T2=parameters[tt][3]
        PD=int(parameters[tt][5]*100)
        LColor1=parameters[tt][6]
        LColor2=parameters[tt][7]
        LColor3=parameters[tt][8]
        prev_x1 = prev_y1 = 0
        

        for x1 in range(1, 450):
            y1 = int(scale_factor_t1*PD * (1-np.exp(-(x1*XscaleTR)/T1)))
            cv2.line(relax_array, (20+prev_x1, 210-prev_y1), (20+x1, 210-y1), (LColor1, LColor2, LColor3), 1)
            prev_x1, prev_y1 = x1, y1
        LM_at_TR = int(scale_factor_t1*PD * (1-np.exp(-(TR/T1))))

        prev_x2 = prev_y2 = x2 = y2 = 0
               
        for x2 in range(1, 450):
            y2 = int(LM_at_TR * np.exp((-(x2*XscaleTE)/T2)))
            cv2.line(relax_array, (20+prev_x2, 460-prev_y2), (x2+20, 460-y2), (LColor1, LColor2, LColor3), 1)
            prev_x2, prev_y2 = x2, y2


def relax_calc_mir(TR, TE, TI):
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1
    if TI ==0:
        TI=1

    XlimitTR = TI+TR
    XscaleTR = XlimitTR/400
    TIloc=int(TI/XscaleTR)
    TRloc=int(TR/XscaleTR)
    scale_factor_t1=1
    scale_factor_t2=1.5

    if TE<100:
        XlimitTE = 120
    else:
        XlimitTE = TE*1.2                                                                                   # 20% more than the TE

    XscaleTE = XlimitTE/450
    TEloc=int(TE/XscaleTE)

    cv2.line(relax_array, (1,290), (512,290), (55,55,55), 1)                                    # X axis line
    cv2.putText(relax_array,'T1 RELAXATION', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(relax_array, (20,160), (470,160), (255,255,255), 1)                                    # X axis line
    cv2.line(relax_array, (20,30), (20,280), (255,255,255), 1)                                      # Y axis line
    cv2.line(relax_array, ((20+TIloc), 40), (20+TIloc, 160), (255,255,255), 1)                      # TI line
    cv2.line(relax_array, (((20+TIloc+TRloc)), 40), ((20+TIloc+TRloc), 280), (255,255,255), 1)        # TR line
    cv2.putText(relax_array, str(int(TI))+'ms', (20+int(TIloc)+5, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TI', (20+int((TIloc))+5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, str(int(TR))+'ms', (20+TIloc+TRloc+5, 180), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TR', (20+TIloc+TRloc+5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    
    cv2.putText(relax_array,'T2 RELAXATION', (200, 315), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(relax_array, (20,460), (470,460), (255,255,255), 1)                                    # RF axis line
    cv2.line(relax_array, (20,300), (20,460), (255,255,255), 1)                                     # RF axis line
    cv2.line(relax_array, (20+TEloc+5, 310), (20+TEloc+5,460), (255,255,255), 1)        # TE line
    cv2.putText(relax_array, str(int(TE))+'ms', (20+TEloc+5, 480), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TE', (20+TEloc+10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    
    cv2.putText(relax_array, 'CSF', (100, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[2][6], parameters[2][7], parameters[2][8]), 1)
    cv2.putText(relax_array, 'Grey Matter', (150, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[3][6], parameters[3][7], parameters[3][8]), 1)
    cv2.putText(relax_array, 'White Matter', (280, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[4][6], parameters[4][7], parameters[4][8]), 1)
    cv2.putText(relax_array, 'Fat', (420, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[5][6], parameters[5][7], parameters[5][8]), 1)


    for tt in range (2,6):
        T1=parameters[tt][2]
        T2=parameters[tt][3]
        PD=int(parameters[tt][5]*100)
        LColor1=parameters[tt][6]
        LColor2=parameters[tt][7]
        LColor3=parameters[tt][8]
        prev_x1 = prev_y1 = 0
        prev_y1_abs = y1_abs = 0
    
        for x1 in range(1, 450):
            TIcomp=int(x1/XscaleTR)
            TRcomp=int(x1-TIcomp)
            y1 = int(scale_factor_t1*PD * (1-2*(np.exp(-(TIcomp*XscaleTR)/T1))*(np.exp(-(TRcomp*XscaleTR)/T1))))
            y1_abs = np.abs(y1)
            cv2.line(relax_array, (20+prev_x1, 160-prev_y1), (20+x1, 160-y1), (LColor1, LColor2, LColor3), 1)
            cv2.line(relax_array, (20+prev_x1, 160-prev_y1_abs), (20+x1, 160-y1_abs), (LColor1, LColor2, LColor3), 1)
            prev_x1, prev_y1 = x1, y1
            prev_y1_abs = y1_abs
        
            if x1 == TIcomp and y1_abs > -3 and y1 < 3:
                y1 = 0
        
        LM_at_TR = int(PD * (1-2*(np.exp(-(TI/T1))*(np.exp(-TR/T1)))))
        
        if TI == int(0.69*T1):
            LM_at_TR = 0
        else:
                LM_at_TR = np.abs(LM_at_TR)

        prev_x2 = prev_y2 = x2 = y2 = 0
               
        for x2 in range(1, 450):
            y2 = int(scale_factor_t2*LM_at_TR * np.exp((-(x2*XscaleTE)/T2)))
            cv2.line(relax_array, (20+prev_x2, 460-prev_y2), (x2+20, 460-y2), (LColor1, LColor2, LColor3), 1)
            prev_x2, prev_y2 = x2, y2


def relax_calc_psir(TR, TE, TI):
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1
    if TI ==0:
        TI=1

    XlimitTR = TI+TR
    XscaleTR = XlimitTR/400
    TIloc=int(TI/XscaleTR)
    TRloc=int(TR/XscaleTR)
    scale_factor_t1=1
    scale_factor_t2=1

    if TE<100:
        XlimitTE = 120
    else:
        XlimitTE = TE*1.2                                                                                   # 20% more than the TE

    XscaleTE = XlimitTE/450
    TEloc=int(TE/XscaleTE)

    cv2.line(relax_array, (1,290), (512,290), (55,55,55), 1)                                    # X axis line
    cv2.putText(relax_array,'T1 RELAXATION', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(relax_array, (20,150), (470,150), (255,255,255), 1)                                    # X axis line
    cv2.line(relax_array, (20,30), (20,270), (255,255,255), 1)                                      # Y axis line
    cv2.line(relax_array, ((20+TIloc), 40), (20+TIloc, 150), (255,255,255), 1)                      # TI line
    cv2.line(relax_array, (((20+TIloc+TRloc)), 40), ((20+TIloc+TRloc), 260), (255,255,255), 1)        # TR line
    cv2.putText(relax_array, str(int(TI))+'ms', (20+int(TIloc)+5, 170), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TI', (20+int((TIloc))+5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, str(int(TR))+'ms', (20+TIloc+TRloc+5, 170), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TR', (20+TIloc+TRloc+5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.putText(relax_array,'T2 RELAXATION', (200, 315), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(relax_array, (20,400), (470,400), (255,255,255), 1)                                    # RF axis line
    cv2.line(relax_array, (20,300), (20,480), (255,255,255), 1)                                     # RF axis line
    cv2.line(relax_array, (20+TEloc+5, 310), (20+TEloc+5,470), (255,255,255), 1)                    # TE line
    cv2.putText(relax_array, str(int(TE))+'ms', (20+TEloc+10, 473), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TE', (20+TEloc+10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.putText(relax_array, 'CSF', (100, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[2][6], parameters[2][7], parameters[2][8]), 1)
    cv2.putText(relax_array, 'Grey Matter', (150, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[3][6], parameters[3][7], parameters[3][8]), 1)
    cv2.putText(relax_array, 'White Matter', (280, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[4][6], parameters[4][7], parameters[4][8]), 1)
    cv2.putText(relax_array, 'Fat', (420, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[5][6], parameters[5][7], parameters[5][8]), 1)


    for tt in range (2,6):
        T1=parameters[tt][2]
        T2=parameters[tt][3]
        PD=int(parameters[tt][5]*100)
        LColor1=parameters[tt][6]
        LColor2=parameters[tt][7]
        LColor3=parameters[tt][8]
        prev_x1 = prev_y1 = 0
        
        for x1 in range(1, 450):
            TIcomp=int(x1/XscaleTR)
            TRcomp=int(x1-TIcomp)
            y1 = int(scale_factor_t1*PD * (1-2*(np.exp(-(TIcomp*XscaleTR)/T1))*(np.exp(-(TRcomp*XscaleTR)/T1))))
            cv2.line(relax_array, (20+prev_x1, 160-prev_y1), (20+x1, 160-y1), (LColor1, LColor2, LColor3), 1)
            prev_x1, prev_y1 = x1, y1
        
        if x1 == TIcomp and y1 > -3 and y1 < 3:
            y1 = 0
        
        LM_at_TR = int(PD * (1-2*(np.exp(-(TI/T1))*(np.exp(-TR/T1)))))

        if TI == int(0.69*T1):
            LM_at_TR = 0

        prev_x2 = prev_y2 = x2 = y2 = 0
               
        for x2 in range(1, 450):
            y2 = int(scale_factor_t2*LM_at_TR * np.exp((-(x2*XscaleTE)/T2)))
            cv2.line(relax_array, (20+prev_x2, 400-prev_y2), (x2+20, 400-y2), (LColor1, LColor2, LColor3), 1)
            prev_x2, prev_y2 = x2, y2

    
def relax_calc_ge(TR, TE, FA):
    if FA ==0:
        FA=1
    if TR ==0:
        TR=1
    if TE ==0:
        TE=1
    XlimitTR = int(TR*1.1)                  # 10% more than the TR
    XscaleTR = XlimitTR/400

    if TE<50:
        XlimitTE = 60
    else:
        XlimitTE = TE*1.2                     # 20% more than the TE
    XscaleTE = XlimitTE/450

    cv2.line(relax_array, (1,236), (512,236), (55,55,55), 1)                                    # X axis line

    cv2.putText(relax_array,'T1 RELAXATION', (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(relax_array, (20,210), (470,210), (255,255,255), 1)                                    # X axis line
    cv2.line(relax_array, (20,30), (20,210), (255,255,255), 1)                                      # Y axis line
    cv2.line(relax_array, (20+int(TR/XscaleTR), 210), (20+int(TR/XscaleTR), 40), (255,255,255), 1)        # TR line
    cv2.putText(relax_array, str(int(TR))+'ms', (20+int(TR/XscaleTR)+5, 230), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TR', (20+int(TR/XscaleTR)+5, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, "FA: "+str(FA)+"deg", (100, 230), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.putText(relax_array,'T2* RELAXATION', (200, 265), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.line(relax_array, (20,460), (470,460), (255,255,255), 1)                                    # RF axis line
    cv2.line(relax_array, (20,260), (20,460), (255,255,255), 1)                                     # RF axis line
    cv2.line(relax_array, (20+int(TE/XscaleTE), 460), (20+int(TE/XscaleTE),272), (255,255,255), 1)        # TE line
    cv2.putText(relax_array, str(int(TE))+'ms', (20+int(TE/XscaleTE)+5, 480), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(relax_array, 'TE', (20+int(TE*XscaleTE)+5, 290), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.putText(relax_array, 'CSF', (100, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[2][6], parameters[2][7], parameters[2][8]), 1)
    cv2.putText(relax_array, 'Grey Matter', (150, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[3][6], parameters[3][7], parameters[3][8]), 1)
    cv2.putText(relax_array, 'White Matter', (280, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[4][6], parameters[4][7], parameters[4][8]), 1)
    cv2.putText(relax_array, 'Fat', (420, 495), cv2.FONT_HERSHEY_PLAIN, 1, (parameters[5][6], parameters[5][7], parameters[5][8]), 1)

    for tt in range (2,6):
        T1=parameters[tt][2]
        T2star=parameters[tt][4]
        PD=int(parameters[tt][5]*100)
        LColor1=parameters[tt][6]
        LColor2=parameters[tt][7]
        LColor3=parameters[tt][8]
        prev_x = prev_y = 0
        
        y_init = int(PD * np.cos(np.deg2rad(FA))) 
    
        for x in range(1, 450):        
            y = int(y_init + (100-y_init)*(1-np.exp(-(x*XscaleTR)/T1)))
            cv2.line(relax_array, (20+prev_x, 210-prev_y), (20+x, 210-y), (LColor1, LColor2, LColor3), 1)
            prev_x, prev_y = x, y
            LM_at_TR = y_init + (100-y_init)*(1-np.exp(-(TR)/T1))
            
        prev_x = prev_y = x = y = 0
               
        for x in range(1, 450):
            y = int(LM_at_TR * np.exp((-(x*XscaleTE)/T2star)))
            cv2.line(relax_array, (20+prev_x, 460-prev_y), (x+20, 460-y), (LColor1, LColor2, LColor3), 1)
            prev_x, prev_y = x, y
    

# SIGNAL TO NOISE RATIO AND SCAN TIME ARE CALCULATED HERE
    
def snr_calc(pseqcse, pseqfse, pseqmir, pseqpsir, pseqcir, pseqfir, iretlvalue,pseqspgr, fastmodenone, fastmodesense, sensefactor, fastmodecsense, csensefactor, acqepi, acqprop, acqspir, spirinterl, slicethkness, fovfreq, acqfreq, acqphase, bwval, nexval, noofslices, faval, trval, teval, tival, etlval, multisliceon, multisliceoff):
    if faval ==0:
        faval=1
    if trval ==0:
        trval=1
    if teval ==0:
        teval=1
    if tival ==0:
        tival=1

    global snr_value
    seq_corr_factor = 1
    fovphase= fovfreq                              # Convert to pixels
    
    if pseqcse==1 or pseqspgr==1:
        etlval=1

    if pseqcse==1 or pseqfse==1 or pseqspgr==1 :
        tival=0

    if pseqmir==1 or pseqpsir==1:
        if pseqcir==1:
            etlval=1
        else:
            etlval = iretlvalue  

    if fastmodesense == 1 or fastmodecsense == 1:
        if fastmodesense == 1:
            sensetimefactor = sensefactor
            sense_snr_corr = 1 / np.sqrt(sensefactor)
        if fastmodecsense == 1: 
            sensetimefactor = csensefactor
            sense_snr_corr = 1 / np.sqrt(csensefactor)
        sense_time_corr = sensetimecorrlist[sensetimefactor][0] 
    else:
        sense_time_corr = 1
        sense_snr_corr = 1

    slices_per_tr = int(trval / teval)
    if multisliceon == 1:
        slices_per_tr = 1
    no_packages = math.ceil(noofslices/slices_per_tr)
    snr_slice_factor = slicethkness / 4; snr_phase_factor = fovfreq/acqphase; snr_freq_factor = fovfreq/acqfreq; snr_bw_factor = bwval/32
    
    snr_value = int(100*sense_snr_corr * seq_corr_factor * (snr_slice_factor * snr_phase_factor * snr_freq_factor * np.sqrt(nexval)) / np.sqrt(snr_bw_factor * snr_phase_factor * snr_freq_factor))

    scan_time = (sense_time_corr * no_packages * nexval * acqphase * (trval + tival)) / (etlval * 1000)

    if acqspir == 1:
        scan_time = scan_time*1.57

    if acqepi == 1:
        scan_time = noofslices

    if acqspir == 1:
        scan_time = noofslices*spirinterl 

    min, sec = divmod(scan_time, 60); hour, min = divmod(min, 60)
    if hour == 0:
            scan_time_formatted = str(int(min)) +' min, '+ str(int(sec)) +' sec'
    elif hour == 0 and min == 0:
            scan_time_formatted = str(int(sec)) +' sec'
    else:
        scan_time_formatted = str(int(hour)) +' hrs, '+ str(int(min)) +' min, '+ str(int(sec)) +' sec'

    calc_snr_message="Calculated SNR            :   "+str(snr_value)+" %" + "\nCalculated Scan Time :   " + str(scan_time_formatted)
    window['-CALCSNR-'].update(calc_snr_message)
    
    param_message = "No of Slices:                  "+str(int(noofslices))+ "\nSlice Thickness:             "+str(int(slicethkness))+" mm  \nField of View:                 "+str(int(fovfreq))+" mm x "+str(int(fovphase))+" mm  \nAcquisition Matrix:       "+str(int(acqfreq))+" x "+str(int(acqphase)) +"\nEcho Train Length:       "+str(int(pseqfse*etlval))+"\nSENSE factor:                 "+str(int(fastmodesense*sensefactor))+"\nCompressed SENSE:     "+str(int(fastmodecsense*csensefactor))+"\nFlip Angle:                        "+str(int(faval))+" deg  \nTR Value:                          "+str(int(trval))+" msec  \nTE Value:                          "+str(int(teval))+" msec  \nTI Value:                            "+str(int(tival))+" msec  \nBandwidth:                      "+str(int(bwval))+" kHz \nNEX Value:                        " +str(int(nexval))
    window['-PARAMETERS-'].update(param_message)




initialize_clean_arrays()


# BEGINNING OF GUI LAYOUT

se_layout = [   [sg.Text("")],
                [sg.Radio("Conventional Spin Echo", group_id=3, default=key_val_dict[5], key = '-PSEQCSE-')], 
                [sg.Radio("Fast Spin Echo", group_id=3, default=key_val_dict[6], key = '-PSEQFSE-'), sg.Text("ETL"), sg.Slider(range = (2, 20), resolution=2, default_value = 1, size =(20,12), orientation = 'horizontal', key = '-ETLVALUE-')],
                [sg.Text("")]
            ]

ir_layout = [   [sg.Text("")],
                [sg.Radio("Magnitude IR", group_id=3, default=key_val_dict[7], key = '-PSEQMIR-')], 
                [sg.Radio("Phase Sensitive IR", group_id=3, default=key_val_dict[10], key = '-PSEQPSIR-')],
                [sg.Radio("Conventional", group_id=11, default=1, key = '-PSEQCIR-')], 
                [sg.Radio("Fast IR", group_id=11, default=0, key = '-PSEQFIR-'), sg.Text("ETL"), sg.Slider(range = (2, 20), resolution=2, default_value = 1, size =(20,12), orientation = 'horizontal', key = '-IRETLVALUE-')],
                [sg.Text("")],
            ]
ge_layout = [   [sg.Text("")],
                [sg.Radio("Spoiled Gradient Echo", group_id=3, default=key_val_dict[11], key = '-PSEQSPGR-')],
                [sg.Text("")],
            ]
ksp_layout = [  
                [sg.Radio("Cartesian", group_id=7, default=key_val_dict[33], key = '-ACQMODECART-')], 
                [sg.Radio("Echo Planar", group_id=7, default=key_val_dict[34], key = '-ACQMODEEPI-'), sg.Slider(range = (1, 2), default_value = 1, size =(34,12), orientation = 'horizontal', key = '-EPISHOTS-')],
                [sg.Radio("Radial", group_id=7, default=key_val_dict[35], key = '-ACQMODERAD-'), sg.Slider(range = (1, 10), default_value = 1, size =(34,12), orientation = 'horizontal', key = '-RADIALSPOKES-')],
                [sg.Radio("Propeller", group_id=7, default=key_val_dict[36], key = '-ACQMODEPROP-'), sg.Slider(range = (10, 20), default_value = 10, size =(34,12), orientation = 'horizontal', key = '-PROPBLADES-')],
                [sg.Radio("Spiral", group_id=7, default=key_val_dict[37], key = '-ACQMODESPIR-'), sg.Slider(range = (1, 4), default_value = 1, size =(34,12), orientation = 'horizontal', key = '-SPIRINTERLEAVES-')],
                [sg.Text("")],
            ]

accel_layout = [ 
                [sg.Radio("None", group_id=4, default=key_val_dict[15], key = '-FASTMODENONE-')], 
                [sg.Radio("Parallel Imaging", group_id=4, default=key_val_dict[16], key = '-FASTMODESENSE-'),
                 sg.Slider(range = (1, 5), default_value = 1, size =(29,12), orientation = 'horizontal', key = '-SENSEFACTOR-')],
                [sg.Radio("Compr. SENSE", group_id=4, default=key_val_dict[17], key = '-FASTMODECSENSE-'), 
                 sg.Slider(range = (1, 5), resolution=1, default_value = 1, size =(23,12), orientation = 'horizontal', key = '-CSENSEFACTOR-')],
            ]

    
tab1_layout = [
                [sg.Frame('Plane of Acquisition', [
                    [sg.Radio('Axial', group_id=5, default=key_val_dict[18], size=(10, 1), key = '-ACQPLANEAX-'),
                     sg.Radio('Coronal', group_id=5, default=key_val_dict[19], size=(10, 1), key = '-ACQPLANECOR-'),
                     sg.Radio('Sagittal', group_id=5, default=key_val_dict[20], size=(10, 1), key = '-ACQPLANESAG-'),
                     ]],font='Any 11', title_color='#1523bd', size=(400, 65))],
                [sg.Frame('Slices', [
                    [sg.Text("No of Slices"), 
                    sg.Slider(range = (1, 30), tick_interval=5, default_value = key_val_dict[21], size =(34,12), orientation = 'horizontal', key = '-NOOFSLICES-')],
                    [sg.Text("Slice Thickness (mm)"), 
                    sg.Slider(range = (1, 5), tick_interval=1, default_value = key_val_dict[22], size =(34,12), orientation = 'horizontal', key = '-SLICETHKNESS-')],
                    [sg.Text("Interslice Gap (mm)"), 
                    sg.Slider(range = (0, 2), tick_interval=0.5, resolution=0.5, default_value = key_val_dict[23], size =(34,12), orientation = 'horizontal', key = '-SLICEGAP-')],
                    [sg.Radio('Multislice Off', group_id=12, default=1, size=(10, 1), key = '-MULTISLICEOFF-'),
                    sg.Radio('Multislice On', group_id=12, default=0, size=(10, 1), key = '-MULTISLICEON-')],
                ],font='Any 11', title_color='#1523bd', size=(400, 265))],
                [sg.Frame('Image Offset', [
                    [sg.Text("R-L Axis (mm)"), 
                    sg.Slider(range = (-100, 100), resolution=5, tick_interval=50, default_value = key_val_dict[24], size =(34,12), orientation = 'horizontal', key = '-RLOFFSET-')],
                    [sg.Text("A-P Axis (mm)"), 
                    sg.Slider(range = (-100, 100), resolution=5, tick_interval=50, default_value = key_val_dict[25], size =(34,12), orientation = 'horizontal', key = '-APOFFSET-')],
                    [sg.Text("H-F Axis (mm)"), 
                    sg.Slider(range = (-100, 100), resolution=5, tick_interval=50, default_value = key_val_dict[26], size =(34,12), orientation = 'horizontal', key = '-HFOFFSET-')],
                ],font='Any 11', title_color='#1523bd', size=(400, 225))],
              ]      

tab2_layout = [
                [sg.Frame('Field of View', [
                [sg.Text("Field of View"), 
                sg.Slider(range = (50, 300), resolution=10, tick_interval=50, default_value = key_val_dict[29], size =(34,12), orientation = 'horizontal', key = '-FOVFREQ-')],
                ],font='Any 11', title_color='#de791b', size=(400, 95))],

                [sg.Frame('Phase Wrap', [
                [sg.Radio("Phase Wrap Allowed", group_id=6, default=key_val_dict[27], key = '-PHASEWRAP-'), sg.Radio("No Phase Wrap", group_id=6, default=key_val_dict[28], key = '-NOPHASEWRAP-')],
                ],font='Any 11', title_color='#de791b', size=(400, 65))],

                [sg.Frame('Acquisition Matrix:', [
                [sg.Text("Frequency Steps"), 
                sg.Slider(range = (64, 512), resolution=8, tick_interval=64, default_value = key_val_dict[31], size =(34,12), orientation = 'horizontal', key = '-ACQFREQ-')],
                [sg.Text("Phase Steps"), 
                sg.Slider(range = (64, 512), resolution=8, tick_interval=64, default_value = key_val_dict[32], size =(34,12), orientation = 'horizontal', key = '-ACQPHASE-')],
                ],font='Any 11', title_color='#de791b', size=(400, 165))],
 
                [sg.Frame('Data Collection', [
                [sg.TabGroup([[sg.Tab('   Kspace Trajectory   ', ksp_layout), sg.Tab(    'Acceleration Technique    ', accel_layout)]])],
                ],font='Any 11', title_color='#de791b', size=(400, 280))],

                [sg.Frame('Signal Averages:', [
                [sg.Text("      NEX Value     "), 
                 sg.Slider(range = (1, 4), tick_interval=0.5, default_value = 1, size =(34,12), orientation = 'horizontal', key = '-NEXVAL-')],
                ],font='Any 11', title_color='#de791b', size=(400, 100))],

              ]

tab3_layout = [ 
                [sg.Frame('Pulse Sequence', [
                [sg.TabGroup([[sg.Tab('   Spin Echo   ', se_layout), sg.Tab('   Inversion Recovery   ', ir_layout), sg.Tab('   Gradient Echo   ', ge_layout)]])]
                ],font='Any 11', title_color='#2c8df5', size=(400, 250))],

                [sg.Frame('Parameters:', [
                [sg.Text("Flip Angle"), 
                sg.Slider(range = (0, 90), tick_interval=20, default_value = key_val_dict[38], size =(34,12), orientation = 'horizontal', key = '-FAVAL-')],
                [sg.Text("TR (millisec)"), 
                sg.Slider(range = (0, 10000), resolution = 10, tick_interval=2000, default_value = key_val_dict[39], size =(34,12), orientation = 'horizontal', key = '-TRVAL-')],
                [sg.Text("TE (millisec)"), 
                sg.Slider(range = (0, 200), tick_interval=50, default_value = key_val_dict[40], size =(34,12), orientation = 'horizontal', key = '-TEVAL-')],
                [sg.Text("TI (millisec)"), 
                sg.Slider(range = (0, 3000), tick_interval=500, default_value = key_val_dict[41], size =(34,12), orientation = 'horizontal', key = '-TIVAL-')],
                [sg.Text("Bandwidth (kHz)"), 
                sg.Slider(range = (16, 64), resolution = 16, tick_interval=16, default_value = key_val_dict[42], size =(34,12), orientation = 'horizontal', key = '-BWVAL-')],
                ],font='Any 11', title_color='#2c8df5', size=(400, 380))],
                 
                ]
    
tab4_layout = [ 
                [sg.Frame('K Space Strategy', [
                [sg.Radio("Full Acq", group_id=10, default=key_val_dict[50], key = '-KSPACEFA-')], 
                [sg.Radio("Partial Fourier", group_id=10, default=key_val_dict[51], key = '-KSPACEPF-')], 
                [sg.Radio("Partial Echo", group_id=10, default=key_val_dict[52], key = '-KSPACEPE-')],
                [sg.Radio("Central k-Space", group_id=10, default=key_val_dict[53], key = '-KSPACEC-')], 
                [sg.Radio("Peripheral k-Space", group_id=10, default=key_val_dict[54], key = '-KSPACEP-')],
                [sg.Radio("Undersampling", group_id=10, default=key_val_dict[55], key = '-KSPACEU-')], 
                [sg.Radio("Zero Filling", group_id=10, default=key_val_dict[56], key = '-KSPACEZ-')],
                ],font='Any 11', title_color='#db3232', size=(400, 250))],

                [sg.Frame('Image Quality & Artifacts:', [
                [sg.Radio("Motion Artifact", group_id=10, default=key_val_dict[58], key = '-IQMOTION-')],
                [sg.Radio("k-Space Spike", group_id=10, default=key_val_dict[60], key = '-IQSPIKE-')],
                [sg.Radio("Zipper Artifact", group_id=10, default=key_val_dict[62], key = '-IQZIPPER-')],
                [sg.Radio("Nyquist Ghosts", group_id=10, default=key_val_dict[63], key = '-IQNYGHOSTS-')],
                [sg.Radio("RF Clipping", group_id=10, default=key_val_dict[64], key = '-IQRFCLIP-')],
                [sg.Radio("DC Offset / Central Point Artifact", group_id=10, default=key_val_dict[65], key = '-IQDCOFFSET-')],
                [sg.Radio("RF Shading Artifact", group_id=10, default=key_val_dict[66], key = '-IQRFSHADE-')],
                ],font='Any 11', title_color='#db3232', size=(400, 260))],

                [sg.Frame('Image Filter', [
                [sg.Radio('Smooth', group_id=9, default=key_val_dict[46], size=(10, 1), key = '-FILTSMOOTH-'),
                 sg.Radio('Normal', group_id=9, default=key_val_dict[47], size=(10, 1), key = '-FILTNORMAL-'),
                 sg.Radio('Sharp', group_id=9, default=key_val_dict[48], size=(10, 1), key = '-FILTSHARP-'),
                ]],font='Any 11', title_color='#db3232', size=(400, 65))], 

                [sg.Frame('Control Sliders', [
                [sg.Text("A"), sg.Slider(range = (1, 100), default_value = 50, size =(33,12), orientation = 'horizontal', enable_events=True, key = '-KSPSLIDERVAL1-')],
                [sg.Text("B"), sg.Slider(range = (1, 100), default_value = 50, size =(33,12), orientation = 'horizontal', disabled=True, enable_events=True, key = '-KSPSLIDERVAL2-')],
                ],font='Any 11', title_color='#db3232', size=(400, 125))],
              ]


tab5_layout = [
                 [sg.Image('', key= '-PSDGRAPH-' , size=(512,512))],
              ]

tab6_layout = [
                 [sg.Image('', key= '-RELAXATIONS-' , size=(512,512))],
              ]

tab7_layout = [
               [sg.Image('', key= '-KSPACETRAJ-' , size=(512,512))],
              ]

tab8_layout = [
               [sg.Image('', key= '-CURRMASK-' , size=(512,512))],
              ]

tab9_layout = [
               [sg.Image('', key= '-CURRIMAGE-' , size=(512,512)), 
                sg.Slider(range = (1, 20), default_value = key_val_dict[75], size =(20,12), orientation = 'vertical', enable_events=True, key = '-CURRSLICE-')],
               [sg.Text("Brightness"), sg.Slider(range = (1, 100), default_value = 50, size =(40,12), orientation = 'horizontal', enable_events=True, key = '-BRIGHTNESS-')],
               [sg.Text("Contrast   "), sg.Slider(range = (1, 100), default_value = 50, size =(40,12), orientation = 'horizontal', enable_events=True, key = '-CONTRAST-')],
              ]

tab10_layout = [
               [sg.Image('', key= '-ZOOMIMAGE-' , size=(512,512)), 
                sg.Slider(range = (-150, 150), resolution = 30, default_value = 0, size =(20,12), orientation = 'vertical', enable_events=True, key = '-ZOOMV-')],
               [sg.Slider(range = (-150, 150), resolution=30, default_value = 0, size =(50,12), orientation = 'horizontal', enable_events=True, key = '-ZOOMH-')],
              ]

tab11_layout = [
               [sg.Frame('PARAMETERS:', title_color='#3278db', layout= 
                [
                 [sg.Text(param_message, key= '-PARAMETERS-', font='Any 12', size=(70,None))]
                ], size=(600, 500))],  
               [sg.Frame('RESULTS:', title_color='#3278db',  layout= 
                [
                 [sg.Text(calc_snr_message, key= '-CALCSNR-', font='Any 12', size=(70,None))]
                ], size=(600, 100))],  
              ]
 


# CONTROL COLUMN AND OUTPUT COLUMN
l_layout = sg.Column([
                    [sg.Frame('USER INPUT', title_color='#1523bd', font='Any 12', layout=
                              [[sg.TabGroup(
                                      [[
                                        sg.Tab('Localization', tab1_layout), 
                                        sg.Tab('Acquisition', tab2_layout), 
                                        sg.Tab('Contrast', tab3_layout),
                                        sg.Tab('Additional', tab4_layout),
                                       ]]
                              )]], size=(400,800)
                     )]
                    ])
          
c_layout = sg.Column([ 
                   [sg.Frame('OUTPUT', title_color='#1523bd', font='Any 12', layout=
                             [[sg.TabGroup(
                                     [[sg.Tab(' Reconstructed Image ', tab9_layout), 
                                       sg.Tab(' Magnified Image ', tab10_layout),
                                       sg.Tab(' Summary of Parameters ', tab11_layout),
                  ]]
               )]], size=(600, 670)
            )],
            [sg.Frame('CONTROL:', title_color='#1523bd', font='Any 12', layout= 
              [[sg.Button("Run", size = (10,2), button_color = ('white','green')), 
               sg.Button("Reset", size = (10,2), button_color = ('white','blue')),
               sg.Button("Quit", size = (10,2), button_color = ('white','red'))]], size=(600, 120))],
         ])

r_layout = sg.Column([ 
                   [sg.Frame('PHYSICS', title_color='#1523bd', font='Any 12', layout=
                             [[sg.TabGroup(
                                     [[sg.Tab(' Pulse Seq. Diagram ', tab5_layout),
                                       sg.Tab(' Relax. Curves ', tab6_layout),
                                       sg.Tab(' k-Space Trajectory ', tab7_layout),
                                       sg.Tab(' k-Space Display' , tab8_layout),
                       ]]
               )]], size=(540, 590)
            )],
            [sg.Frame('MESSAGE:', title_color='#1523bd', font='Any 12', layout= 
               [
                [sg.Text(message1_text, key= '-MESSAGE1-', font='Any 11', size=(70,None))],
                ], size=(540, 200))],    
         ])

    
# MAIN WINDOW    
layout = [[l_layout, c_layout, r_layout]],

sg.set_options(text_justification='left')

window = sg.Window('The MRI Simulator', layout, font='arial 11', default_element_size=(12,1), resizable=True, finalize=True)

while True:
    event, values = window.read(timeout = 50)
    if event == 'Quit'  or event == sg.WIN_CLOSED:
        break
    if event == 'Run':
        initialize_clean_arrays()

        create_brain_map(values['-PSEQCSE-' ],values['-PSEQFSE-'],values['-ETLVALUE-'],
                         values['-PSEQMIR-'], values['-PSEQPSIR-'],values['-PSEQFIR-'], values['-IRETLVALUE-'], values['-PSEQSPGR-'],
                         values['-FASTMODENONE-'],values['-FASTMODESENSE-'],values['-SENSEFACTOR-'],values['-FASTMODECSENSE-'],values['-CSENSEFACTOR-'],
                         values['-ACQPLANEAX-'],values['-ACQPLANECOR-'],values['-ACQPLANESAG-'],values['-NOOFSLICES-'],values['-SLICETHKNESS-'],values['-SLICEGAP-'],
                         values['-RLOFFSET-'],values['-APOFFSET-'],values['-HFOFFSET-'],
                         values['-FOVFREQ-'],values[ '-PHASEWRAP-'],values['-NOPHASEWRAP-'],values['-ACQFREQ-'],values['-ACQPHASE-'],
                         values['-ACQMODECART-'],values['-ACQMODEEPI-'],values['-EPISHOTS-'],values['-ACQMODERAD-'],values['-RADIALSPOKES-'],values['-ACQMODEPROP-'],values['-PROPBLADES-'],values['-ACQMODESPIR-'],values['-SPIRINTERLEAVES-'],
                         values['-FAVAL-'],values['-TRVAL-'],values['-TEVAL-'],values['-TIVAL-'],values['-BWVAL-'],
                         values['-NEXVAL-'],values['-FILTSMOOTH-'],values['-FILTNORMAL-'],values['-FILTSHARP-'],
                         values['-KSPACEFA-'],values['-KSPACEPF-'],values['-KSPACEPE-'],values['-KSPACEC-'],values['-KSPACEP-'],values['-KSPACEU-'],values['-KSPACEZ-'],
                         values['-IQMOTION-'],values['-IQSPIKE-'],values['-IQZIPPER-'],values['-IQNYGHOSTS-'],values['-IQRFCLIP-'],values['-IQDCOFFSET-'],values['-IQRFSHADE-'],
                         values['-KSPSLIDERVAL1-'],values['-KSPSLIDERVAL2-'],
                         values['-CURRSLICE-'],values['-BRIGHTNESS-'],values['-CONTRAST-']
                         )

        if values['-PSEQCSE-']==1 or values['-PSEQFSE-']==1:
            graphPSD = sg.Graph(canvas_size=(512, 512), graph_bottom_left=(0,0), graph_top_right=(512, 512), background_color='black')
            psd_calc_se(values['-TRVAL-'],values['-TEVAL-'],values['-ETLVALUE-'], values['-PSEQFSE-'])            
            relax_calc_se(values['-TRVAL-'],values['-TEVAL-'])
            draw = window['-PSDGRAPH-']
            draw = window['-RELAXATIONS-']
            
        if values['-PSEQMIR-']==1:
            graphPSD = sg.Graph(canvas_size=(512, 512), graph_bottom_left=(0,0), graph_top_right=(512, 512), background_color='black')
            psd_calc_ir(values['-TRVAL-'],values['-TEVAL-'],values['-TIVAL-'],values['-ETLVALUE-'])
            relax_calc_mir(values['-TRVAL-'],values['-TEVAL-'],values['-TIVAL-'])
            draw = window['-PSDGRAPH-']
            draw = window['-RELAXATIONS-']

        if  values['-PSEQPSIR-']==1:
            graphPSD = sg.Graph(canvas_size=(512, 512), graph_bottom_left=(0,0), graph_top_right=(512, 512), background_color='black')
            psd_calc_ir(values['-TRVAL-'],values['-TEVAL-'],values['-TIVAL-'],values['-ETLVALUE-'])
            relax_calc_psir(values['-TRVAL-'],values['-TEVAL-'],values['-TIVAL-'])
            draw = window['-PSDGRAPH-']
            draw = window['-RELAXATIONS-']
            
        if values['-PSEQSPGR-']==1 :
            graphPSD = sg.Graph(canvas_size=(512, 512), graph_bottom_left=(0,0), graph_top_right=(512, 512), background_color='black')
            psd_calc_ge(values['-FAVAL-'],values['-TRVAL-'],values['-TEVAL-'])
            relax_calc_ge(values['-TRVAL-'],values['-TEVAL-'],values['-FAVAL-'])
            draw = window['-PSDGRAPH-']
            draw = window['-RELAXATIONS-']

        snr = snr_calc(values['-PSEQCSE-' ],values['-PSEQFSE-'], values['-PSEQMIR-'], values['-PSEQPSIR-'],values['-PSEQCIR-'],values['-PSEQFIR-'],values['-IRETLVALUE-'],values['-PSEQSPGR-'],
                 values['-FASTMODENONE-'],values['-FASTMODESENSE-'],values['-SENSEFACTOR-'],values['-FASTMODECSENSE-'],values['-CSENSEFACTOR-'],values['-ACQMODEEPI-'], values['-ACQMODEPROP-'], values['-ACQMODESPIR-'], values['-SPIRINTERLEAVES-'],  
                 values['-SLICETHKNESS-'],values['-FOVFREQ-'],values['-ACQFREQ-'],values['-ACQPHASE-'],values['-BWVAL-'],values['-NEXVAL-'],
                 values['-NOOFSLICES-'],values['-FAVAL-'],values['-TRVAL-'],values['-TEVAL-'],values['-TIVAL-'],values['-ETLVALUE-'],
                 values['-MULTISLICEOFF-'],values['-MULTISLICEON-']
                 )
    
    elif event == 'Reset':
       reset_values()

    update_image(values['-NOOFSLICES-'],values['-CURRSLICE-'],values['-BRIGHTNESS-'],values['-CONTRAST-'], values['-PSEQCSE-'],values['-PSEQFSE-'],values['-PSEQMIR-'],values['-PSEQPSIR-'],values['-PSEQSPGR-'],
                 values['-KSPACEFA-'], values['-KSPACEPF-'], values['-KSPACEPE-'], values['-KSPACEC-'], values['-KSPACEP-'], values['-KSPACEU-'], values['-KSPACEZ-'], 
                 values['-IQMOTION-'], values['-IQSPIKE-'], values['-IQZIPPER-'], 
                 values['-IQNYGHOSTS-'], values['-IQRFCLIP-'], values['-IQDCOFFSET-'], values['-IQRFSHADE-'],
                 values['-ACQMODECART-'], values['-ACQMODEPROP-'], values['-ACQMODEEPI-'], values['-FILTSMOOTH-'], values['-FILTNORMAL-'], values['-FILTSHARP-'], values['-FILTSHARP-'],
                 values['-ZOOMH-'],values['-ZOOMV-'], snr_value
                 )
window.close()

