
import numpy as np


for i in range(1, 2):
    folder = 'lens_search'
    add = str(i)

    images = np.load('data/' + folder + '/specImgs' + add + '.npz')['arr_0']
    spectra = np.load('data/' + folder + '/spectra' + add + '.npz', allow_pickle=True)['arr_0']
    loglams = np.load('data/' + folder + '/loglam' + add + '.npz', allow_pickle=True)['arr_0']

    max_start = 0
    inds = []
    for i in range(len(loglams)):
        if np.min(loglams[i]) > 3.59:
            inds.append(i)
            continue
        if np.max(loglams[i]) < 3.95:
            inds.append(i)
            continue

        if np.min(loglams[i]) > max_start:
            max_start = np.min(loglams[i])

    images = np.delete(images, inds, axis=0)
    spectra = np.delete(spectra, inds, axis=0)
    loglams = np.delete(loglams, inds, axis=0)

    min_end = 6
    for i in range(len(loglams)):
        if np.max(loglams[i]) < min_end:
            min_end = np.max(loglams[i])        

    for i in range(len(loglams)):
        start = np.argwhere(loglams[i] == max_start).flatten()[0]
        end = np.argwhere(loglams[i] == min_end).flatten()[0]

        start = np.argwhere(loglams[i] == 3.5901).flatten()[0]
        end = np.argwhere(loglams[i] == 3.9499).flatten()[0]

        loglams[i] = loglams[i][start:end]
        spectra[i] = spectra[i][start:end]

    np.savez('data/' + folder + '/spectra_uniform' + add + '.npz', spectra)
    np.savez('data/' + folder + '/loglam_uniform' + add + '.npz', loglams)
    np.savez('data/' + folder + '/specImgs_uniform' + add + '.npz', images)