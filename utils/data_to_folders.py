
import numpy as np


for add in range(1, 7):
    folder = 'galaxies'

    images = np.load('data/' + folder + '/specImgs_uniform' + str(add) + '.npz')['arr_0']
    spectra = np.load('data/' + folder + '/spectra_uniform' + str(add) + '.npz', allow_pickle=True)['arr_0']
    indices = np.loadtxt('data/' + folder + '/specInds_uniform' + str(add) + '.txt')

    spectra = [spectra[i][None] for i in range(len(spectra))]
    spectra = np.concatenate(spectra, axis=0)

    print(images.shape, spectra.shape)

    mins = np.min(spectra, axis=1)
    maxes = np.max(spectra, axis=1)

    inds = np.where((maxes[:, None] - mins[:, None])==0)

    images = np.delete(images, inds, axis=0)
    spectra = np.delete(spectra, inds, axis=0)
    indices = np.delete(indices, inds, axis=0)

    print(images.shape)

    split = 'train'
    for i in range(len(images)):
        if add == 1:
            split = 'val'

        np.save('data/' + folder + '/data_files/' + split + '/images/' + str((add - 1) * 100000 + i), images[i])
        np.save('data/' + folder + '/data_files/' + split + '/spectra/' + str((add - 1) * 100000 + i), spectra[i])
        np.save('data/' + folder + '/data_files/' + split + '/indices/' + str((add - 1) * 100000 + i), indices[i])