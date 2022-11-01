
import numpy as np


for i in range(6):
    folder = 'galaxies'
    start = i * 4
    end = (i + 1) * 4
    name = str(i + 1)

    print(start, end)

    images = np.load('data/' + folder + '/downloaded_files/specImgs' + str(start) + '.npz')['arr_0']
    spectra = np.load('data/' + folder + '/downloaded_files/spectra' + str(start) + '.npz', allow_pickle=True)['arr_0']
    loglams = np.load('data/' + folder + '/downloaded_files/loglam' + str(start) + '.npz', allow_pickle=True)['arr_0']
    indices = np.loadtxt('data/' + folder + '/downloaded_files/specInds' + str(start) + '.txt')

    for num in range(start + 1, end):
        images2 = np.load('data/' + folder + '/downloaded_files/specImgs' + str(num) + '.npz')['arr_0']
        spectra2 = np.load('data/' + folder + '/downloaded_files/spectra' + str(num) + '.npz', allow_pickle=True)['arr_0']
        loglams2 = np.load('data/' + folder + '/downloaded_files/loglam' + str(num) + '.npz', allow_pickle=True)['arr_0']
        indices2 = np.loadtxt('data/' + folder + '/downloaded_files/specInds' + str(num) + '.txt')

        images = np.concatenate((images, images2), axis=0)
        spectra = np.concatenate((spectra, spectra2), axis=0)
        loglams = np.concatenate((loglams, loglams2), axis=0)
        indices = np.concatenate((indices, indices2), axis=0)
        print(images.shape, spectra.shape, loglams.shape, indices.shape)

    np.savez('data/' + folder + '/specImgs' + name + '.npz', images)
    np.savez('data/' + folder + '/spectra' + name + '.npz', spectra)
    np.savez('data/' + folder + '/loglam' + name + '.npz', loglams)
    np.savetxt('data/' + folder + '/specInds' + name + '.txt', indices)
