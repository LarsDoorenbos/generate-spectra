
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd

from scipy.interpolate import CubicSpline
from astropy.convolution import convolve, Gaussian1DKernel

g = Gaussian1DKernel(stddev=2)


def find_loc(catalog, obj_id):
    start_diff = 10000000
    i = np.where(np.abs(catalog["specObjID"] - obj_id) < start_diff)
    
    if len(i[0]) == 0:
        print('No match found')
        return None

    if len(i[0]) > 1:
        raise ValueError('More than 1 possible match found')

    i = i[0][0]
    return i


loglams = np.load('data/5band/downloaded_files/loglam0.npz', allow_pickle=True)['arr_0'][0]

start = np.argwhere(loglams == 3.5901).flatten()[0]
end = np.argwhere(loglams == 3.9499).flatten()[0]

BASE_PATH = 'data/5band/eval_9010'

img_file_list = sorted(glob.glob(BASE_PATH + "/test_set/images/*.npy"))
spec_file_list = sorted(glob.glob(BASE_PATH + "/test_set/spectra/*.npy"))
ind_file_list = sorted(glob.glob(BASE_PATH + "/test_set/indices/*.npy"))

ind_order = np.loadtxt('inds_5band_9010.txt').astype(int)

test_dataset = list(zip(img_file_list, spec_file_list, ind_file_list))

loglam = np.load('data/5band/loglam_uniform1.npz', allow_pickle=True)['arr_0'][0]
orig_wavelengths = 10 ** loglam

trimmed_orig_wavelengths = orig_wavelengths[orig_wavelengths < 8500]
trimmed_orig_wavelengths = trimmed_orig_wavelengths[trimmed_orig_wavelengths > 4000]

wavelengths = np.arange(np.round(np.min(trimmed_orig_wavelengths)), np.round(np.max(trimmed_orig_wavelengths)))

for idx, orig_ind in enumerate(ind_order):    
    os.makedirs(os.path.join(BASE_PATH, 'output', str(idx)), exist_ok=True)

    if idx % 10 == 0:
        print(idx)

    spec_obj_id, redshift = np.load(BASE_PATH + "/test_set/indices/" + str(orig_ind) + ".npy")
    spectrum = np.load(BASE_PATH + "/test_set/spectra/" + str(orig_ind) + ".npy")
    img = np.load(BASE_PATH + "/test_set/images/" + str(orig_ind) + ".npy")

    corrected_spectrum = spectrum
    corrected_spectrum = corrected_spectrum[orig_wavelengths < 8500]
    corrected_spectrum = corrected_spectrum[orig_wavelengths[orig_wavelengths < 8500] > 4000]

    cs = CubicSpline(trimmed_orig_wavelengths, corrected_spectrum)
    interpolated_spectrum = cs(wavelengths)

    smoothed_spectrum = convolve(interpolated_spectrum, g)

    norm_factor = np.mean(smoothed_spectrum[int(6900 - np.round(np.min(wavelengths))):int(6950 - np.round(np.min(wavelengths)))])
    normalized_spectrum = smoothed_spectrum / norm_factor

    all_catalog = pd.read_csv('data/5band/005-015catalog.csv')
    i = find_loc(all_catalog, spec_obj_id)

    ra = all_catalog.loc[i]['ra']
    dec = all_catalog.loc[i]['dec']
    obj_id = all_catalog.loc[i]['objID']
    z_err = all_catalog.loc[i]['zerr']
    vel_disp = all_catalog.loc[i]['velDisp']
    vel_disp_err = all_catalog.loc[i]['velDispErr']
    mag_u = all_catalog.loc[i]['cmodelMag_u']
    mag_g = all_catalog.loc[i]['cmodelMag_g']
    mag_r = all_catalog.loc[i]['cmodelMag_r']
    mag_i = all_catalog.loc[i]['cmodelMag_i']
    mag_z = all_catalog.loc[i]['cmodelMag_z']

    i = find_loc(all_catalog, spec_obj_id)

    plate = all_catalog.loc[i]['plate']
    mjd = all_catalog.loc[i]['mjd']
    fiberID = all_catalog.loc[i]['fiberid']

    try:
        hdul = fits.open('http://dr16.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/' + str(plate).rjust(4, '0') + '/spec-' + str(plate).rjust(4, '0') + '-' + str(mjd) + '-' + str(fiberID).rjust(4, '0') + '.fits')
        spec = np.array(hdul[1].data)
        ivar = np.array([spec[i][2] for i in range(len(spec))])
        hdul.close()
    except:
        raise ValueError('Download of spectrum failed')
    
    ivar = ivar[start:end]

    generated_spectrum = np.load(os.path.join(BASE_PATH, 'generations', str(idx) + '/0.npy'))[0]

    i = find_loc(all_catalog, spec_obj_id)

    if i == None:
        print('no mass')
        log_mass_maraston = np.nan
        log_mass_conroy_dust = np.nan
        log_mass_conroy_no_dust = np.nan
    else:
        log_mass_maraston = all_catalog.loc[i]['logMass_Maraston09']
        log_mass_conroy_dust = all_catalog.loc[i]['logMass_ConroyDust']
        log_mass_conroy_no_dust = all_catalog.loc[i]['logMass_ConroyNoDust']

    i = find_loc(all_catalog, spec_obj_id)

    if i == None:
        print('no zoo')
        p_mg = np.nan
        p_el_debiased = np.nan
        p_cs_debiased = np.nan
        spiral = np.nan
        elliptical = np.nan
        uncertain = np.nan
    else:
        p_mg = all_catalog.loc[i]['p_mg']
        p_el_debiased = all_catalog.loc[i]['p_el_debiased']
        p_cs_debiased = all_catalog.loc[i]['p_cs_debiased']
        spiral = all_catalog.loc[i]['spiral']
        elliptical = all_catalog.loc[i]['elliptical']
        uncertain = all_catalog.loc[i]['uncertain']
    
    prim_hdu = fits.PrimaryHDU()

    col4 = fits.Column(name='ra', format='E', array=[ra])
    col5 = fits.Column(name='dec', format='E', array=[dec])
    col6 = fits.Column(name='obj_id', format='E', array=[obj_id])
    col7 = fits.Column(name='redshift', format='E', array=[redshift])
    col8 = fits.Column(name='z_err', format='E', array=[z_err])
    col9 = fits.Column(name='vel_disp', format='E', array=[vel_disp])
    col10 = fits.Column(name='vel_disp_err', format='E', array=[vel_disp_err])
    col11 = fits.Column(name='mag_u', format='E', array=[mag_u])
    col12 = fits.Column(name='mag_g', format='E', array=[mag_g])
    col13 = fits.Column(name='mag_r', format='E', array=[mag_r])
    col14 = fits.Column(name='mag_i', format='E', array=[mag_i])
    col15 = fits.Column(name='mag_z', format='E', array=[mag_z])
    col16 = fits.Column(name='plate', format='E', array=[plate])
    col17 = fits.Column(name='mjd', format='E', array=[mjd])
    col18 = fits.Column(name='fiberID', format='E', array=[fiberID])
    col19 = fits.Column(name='log_mass_maraston', format='E', array=[log_mass_maraston])
    col20 = fits.Column(name='log_mass_conroy_dust', format='E', array=[log_mass_conroy_dust])
    col21 = fits.Column(name='log_mass_conroy_no_dust', format='E', array=[log_mass_conroy_no_dust])
    col22 = fits.Column(name='p_mg', format='E', array=[p_mg])
    col23 = fits.Column(name='p_el_debiased', format='E', array=[p_el_debiased])
    col24 = fits.Column(name='p_cs_debiased', format='E', array=[p_cs_debiased])
    col25 = fits.Column(name='spiral', format='E', array=[spiral])
    col26 = fits.Column(name='elliptical', format='E', array=[elliptical])
    col27 = fits.Column(name='uncertain', format='E', array=[uncertain])

    cols = fits.ColDefs([col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27])

    obj_inf_hdu = fits.BinTableHDU.from_columns(cols)

    col1 = fits.Column(name='orig_wavelength', format='J', array=orig_wavelengths)
    col2 = fits.Column(name='orig_spectrum', format='E', array=spectrum)
    col3 = fits.Column(name='ivar', format='E', array=ivar)

    cols = fits.ColDefs([col1, col2, col3])

    orig_spectrum_hdu = fits.BinTableHDU.from_columns(cols)

    col1 = fits.Column(name='wavelength', format='J', array=wavelengths)
    col2 = fits.Column(name='gt_spectrum', format='E', array=normalized_spectrum)
    col3 = fits.Column(name='generated_spectrum', format='E', array=generated_spectrum)

    cols = fits.ColDefs([col1, col2, col3])

    generated_spectrum_hdu = fits.BinTableHDU.from_columns(cols)

    hdul = fits.HDUList([prim_hdu, obj_inf_hdu, orig_spectrum_hdu, generated_spectrum_hdu])
    hdul.writeto(os.path.join(BASE_PATH, 'output', str(idx), str(idx) + '.fits'), overwrite=True)

    plt.figure()
    plt.plot(wavelengths, normalized_spectrum, label='spectrum', lw=0.8, alpha=0.85)
    plt.plot(wavelengths, generated_spectrum, label='generated spectrum', lw=0.8, alpha=0.85)
    plt.legend()
    plt.savefig(os.path.join(BASE_PATH, 'output', str(idx), str(idx) + '.png'))
    plt.close()

    plt.figure()
    plt.imshow(img[:, :, 1:-1])
    plt.savefig(os.path.join(BASE_PATH, 'output', str(idx), str(idx) + '_img.png'))
    plt.close()