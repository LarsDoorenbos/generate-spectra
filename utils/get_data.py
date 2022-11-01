
import imageio
import numpy as np
from tqdm import tqdm
import pandas as pd
from astropy.io import fits

name = 'galaxies'
num = 0
offset = num * 25000
size = 25000

catalog = pd.read_csv('data/galaxies/3mGalaxies.csv')	
print(len(catalog))

catalog = catalog.sample(frac=1, random_state=1)
catalog = catalog[offset:offset+size]

mjd = catalog['mjd'].values
plate = catalog['plate'].values
fiber = catalog['fiberID'].values
ra = catalog['ra'].values
dec = catalog['dec'].values
specIDs = catalog['specObjID'].values
instruments = catalog['instrument'].values
run2ds = catalog['run2d'].values

instruments = np.where(instruments == 'SDSS', 'sdss', 'eboss')

imageList = []
specList = []
loglamList = []

coords = []
err = 0

for idx in tqdm(range(len(ra))):
    url = 'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=' + str(ra[idx]) + '&dec=' + str(dec[idx]) + '&scale=0.4&height=64&width=64'
    for i in range(1):  # retry loop
        try:
            im = imageio.imread(url)
            break  # On success, stop retry.
        except:
            print('Download of image failed')
            err = 1
    for i in range(1):  # retry loop
        if err == 0:
            try:
                hdul = fits.open('http://dr16.sdss.org/sas/dr16/' + instruments[idx] + '/spectro/redux/' + str(run2ds[idx]) + '/spectra/lite/' + str(plate[idx]).rjust(4, '0') + '/spec-' + str(plate[idx]).rjust(4, '0') + '-' + str(mjd[idx]) + '-' + str(fiber[idx]).rjust(4, '0') + '.fits')
                spec = np.array(hdul[1].data)
                loglam = np.array([spec[i][1] for i in range(len(spec))])
                spec = np.array([spec[i][0] for i in range(len(spec))])
                
                imageList.append(im)
                specList.append(spec)
                loglamList.append(loglam)
                coords.append(specIDs[idx])

                break
            except:
                print('Download of spectrum failed')
                print('http://dr16.sdss.org/sas/dr16/' + instruments[idx] + '/spectro/redux/' + str(run2ds[idx]) + '/spectra/lite/' + str(plate[idx]).rjust(4, '0') + '/spec-' + str(plate[idx]).rjust(4, '0') + '-' + str(mjd[idx]) + '-' + str(fiber[idx]).rjust(4, '0') + '.fits')

    err = 0

imageList = np.array(imageList)
specList = np.array(specList)

print(imageList.shape, len(coords), specList.shape)

np.savez('data/' + name + '/downloaded_files/specImgs' + str(num) + '.npz', imageList)
np.savez('data/' + name + '/downloaded_files/spectra' + str(num) + '.npz', specList)
np.savez('data/' + name + '/downloaded_files/loglam' + str(num) + '.npz', loglamList)
np.savetxt('data/' + name + '/downloaded_files/specInds' + str(num) + '.txt', coords)