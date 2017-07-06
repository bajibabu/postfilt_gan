import numpy as np
# To run background with matplotlib
# use these two lines
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt

def plot_feats(actual_feats, predicted_feats, generated_feats, epoch, index, checkpoint_dir):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(actual_feats)
    plt.subplot(1,3,2)
    plt.imshow(generated_feats)
    plt.subplot(1,3,3)
    plt.imshow(predicted_feats)
    plt.savefig('{}/figures/pulses_epoch{}_index{}.png'.format(checkpoint_dir, epoch, index))
    plt.close()

def read_binary_file(file, dim=1):
    f = open(file, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    assert data.shape[0] % dim == 0.
    data = data.reshape(-1, dim)
    return data, data.shape[0]
