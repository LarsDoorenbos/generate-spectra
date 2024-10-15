
import os, platform

import numpy as np
import pandas as pd

folder = '5band'

os.makedirs('data/' + folder + '/data_files/005-015/images/', exist_ok=True)
os.makedirs('data/' + folder + '/data_files/005-015/spectra/', exist_ok=True)
os.makedirs('data/' + folder + '/data_files/005-015/indices/', exist_ok=True)

sdss = np.load('data/5band/sdss.npz')

images = sdss["cube"]
labels = sdss["labels"]
sdss_specobjids = labels["specObjID"]

for add in range(1):
    spectra = np.load('data/' + folder + '/spectra_uniform' + str(add) + '.npz', allow_pickle=True)['arr_0']
    indices = np.loadtxt('data/' + folder + '/specInds_uniform' + str(add) + '.txt')
    catalog = pd.read_csv('data/5band/005-015catalog.csv')	

    spectra = [spectra[i][None] for i in range(len(spectra))]
    spectra = np.concatenate(spectra, axis=0)

    print(spectra.shape)
    print(images.shape)

    mins = np.min(spectra, axis=1)
    maxes = np.max(spectra, axis=1)

    inds = np.where((maxes[:, None] - mins[:, None])==0)

    spectra = np.delete(spectra, inds, axis=0)
    indices = np.delete(indices, inds, axis=0)

    print(len(catalog))

    redshift_catalog = catalog['z'].to_numpy()
    objid_catalog = catalog['objID'].to_numpy()
    specobjid_catalog = catalog['specObjID'].to_numpy()

    redshifts = []
    image_list = []
    for num, i in enumerate(indices):
        # Find match in catalog
        start_diff = 10000000
        idx = np.where(np.abs(catalog["specObjID"] - i) < start_diff)
        
        while len(idx[0]) == 0:
            start_diff *= 5
            idx = np.where(np.abs(catalog.loc[i]["specObjID"] - i) < start_diff)

        if len(idx[0]) > 1:
            print(idx[0])
            print(catalog['specObjID'].to_numpy()[idx[0]])
            raise ValueError('More than 1 possible match found')

        idx = idx[0][0]

        redshifts.append(redshift_catalog[idx])

        # Find match in npz data cube
        start_diff = 10000000
        idx = np.where(np.abs(sdss_specobjids - i) < start_diff)

        while len(idx[0]) == 0:
            start_diff *= 5
            idx = np.where(np.abs(sdss_specobjids - i) < start_diff)

        if len(idx[0]) > 1:
            print(idx[0])
            print(sdss_specobjids[idx[0]])
            raise ValueError('More than 1 possible match found')
        
        image_list.append(images[idx[0][0]])

    image_list = np.array(image_list)

    print(image_list.shape)

    for i in range(len(spectra)):
        split = '005-015'

        np.save('data/' + folder + '/data_files/' + split + '/images/' + str((add) * 100000 + i), image_list[i])
        np.save('data/' + folder + '/data_files/' + split + '/spectra/' + str((add) * 100000 + i), spectra[i])
        np.save('data/' + folder + '/data_files/' + split + '/indices/' + str((add) * 100000 + i), [indices[i], redshifts[i]])
