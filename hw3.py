from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)

def get_covariance(dataset):
    return 1/(dataset.shape[0]-1)*np.dot(np.transpose(dataset),dataset)

def get_eig(S, m):
    Lambda, U = eigh(S,subset_by_index=[S.shape[0]-m, S.shape[0]-1])
    l = np.diag(np.flip(Lambda,axis=0))
    u = U[:,np.flip(np.argsort(Lambda))]
    return l,u

def get_eig_prop(S, prop):
    s = sum(eigh(S,eigvals_only=True))
    Lambda, U = eigh(S,subset_by_value=[prop*s, np.inf])
    l = np.diag(np.flip(np.sort(Lambda)))
    u = U[:,np.flip(np.argsort(Lambda))]
    return l, u

def project_image(image, U):
    return np.dot(np.dot(np.transpose(U),image),np.transpose(U))

def display_image(orig, proj):
    f, (ax1, ax2) = plt.subplots(1, 2)
    col1 = ax1.imshow(np.transpose(orig.reshape(32,32)),aspect='equal')
    col2 = ax2.imshow(np.transpose(proj.reshape(32,32)),aspect='equal')
    ax1.set_title("Original")
    ax2.set_title("Projection")
    f.colorbar(col1, ax=ax1)
    f.colorbar(col2, ax=ax2)
    return plt.show()
