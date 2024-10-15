
import numpy as np
from tqdm import tqdm
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt


name = '5band'
num = 0
offset = num * 50000
size = 50000

catalog = pd.read_csv('data/5band/spectraGeneration_cavuoti_1.csv')	
print(len(catalog))

redshifts = catalog['z'].to_numpy()

plt.hist(redshifts, bins=100)
plt.savefig('redshifts.png')

zcut = redshifts[redshifts > 0.05]
zcut = zcut[zcut < 0.15]

print(len(zcut))

catalog = catalog[catalog['z'] > 0.05]
catalog = catalog[catalog['z'] < 0.15]

# save catalog
catalog.to_csv('data/' + name + '/005-015catalog.csv')

print(len(catalog))

catalog = catalog[offset:offset+size]

mjd = catalog['mjd'].values
plate = catalog['plate'].values
fiber = catalog['fiberid'].values
ra = catalog['ra'].values
dec = catalog['dec'].values
specIDs = catalog['specObjID'].values
instruments = catalog['instrument'].values
run2ds = catalog['run2d'].values

instruments = np.where(instruments == 'SDSS', 'sdss', 'eboss')

specList = []
loglamList = []

coords = []
err = 0

for idx in tqdm(range(len(ra))):
    url = 'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=' + str(ra[idx]) + '&dec=' + str(dec[idx]) + '&scale=0.4&height=64&width=64'
    for i in range(1):  # retry loop
        if err == 0:
            try:
                hdul = fits.open('http://dr16.sdss.org/sas/dr16/' + instruments[idx] + '/spectro/redux/' + str(run2ds[idx]) + '/spectra/lite/' + str(plate[idx]).rjust(4, '0') + '/spec-' + str(plate[idx]).rjust(4, '0') + '-' + str(mjd[idx]) + '-' + str(fiber[idx]).rjust(4, '0') + '.fits')
                spec = np.array(hdul[1].data)
                loglam = np.array([spec[i][1] for i in range(len(spec))])
                spec = np.array([spec[i][0] for i in range(len(spec))])
                                
                specList.append(spec)
                coords.append(specIDs[idx])
                loglamList.append(loglam)

                break
            except Exception as e:
                print(e)
                print('Download of spectrum failed')
                print('http://dr16.sdss.org/sas/dr16/' + instruments[idx] + '/spectro/redux/' + str(run2ds[idx]) + '/spectra/lite/' + str(plate[idx]).rjust(4, '0') + '/spec-' + str(plate[idx]).rjust(4, '0') + '-' + str(mjd[idx]) + '-' + str(fiber[idx]).rjust(4, '0') + '.fits')

    err = 0

specList = np.array(specList)

print(len(coords), specList.shape, specList[0].shape)

np.savez('data/' + name + '/downloaded_files/spectra' + str(num) + '.npz', specList)
np.savetxt('data/' + name + '/downloaded_files/specInds' + str(num) + '.txt', coords)
np.savez('data/' + name + '/downloaded_files/loglam' + str(num) + '.npz', loglamList)