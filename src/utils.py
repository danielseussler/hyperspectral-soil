import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from umap import UMAP


class SpectralCurveFiltering():
    """
    Create a histogram (a spectral curve) of a 3D cube, using the merge_function
    to aggregate all pixels within one band. The return array will have
    the shape of [CHANNELS_COUNT]
    """

    def __init__(self, merge_function=np.mean):
        self.merge_function = merge_function

    def __call__(self, sample: np.ndarray):
        return self.merge_function(sample, axis=(1, 2))


def load_data(directory: str):
    """Load each cube, reduce its dimensionality and append to array.

    Args:
        directory (str): Directory to either train or test set
    Returns:
        [type]: A list with spectral curve for each sample.
    """

    data = []
    filtering = SpectralCurveFiltering()
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )
    for file_name in all_files:
        with np.load(file_name) as npz:
            arr = np.ma.MaskedArray(**npz)
        arr = filtering(arr)
        data.append(arr)
    return np.array(data)


def load_gt(file_path: str):
    """Load labels for train set from the ground truth file.
    Args:
        file_path (str): Path to the ground truth .csv file.
    Returns:
        [type]: 2D numpy array with soil properties levels
    """

    gt_file = pd.read_csv(file_path)
    labels = gt_file[["P", "K", "Mg", "pH"]].values
    return labels


def scale_by_mean(arr1, arr2, arr3=None):
    """Scale arrays by the column means of the first array passed."""
    mean_arr1 = np.mean(arr1, axis=0)
    scaled_arr1 = arr1 / mean_arr1
    scaled_arr2 = arr2 / mean_arr1

    if arr3 is not None:
        scaled_arr3 = arr3 / mean_arr1
        return scaled_arr1, scaled_arr2, scaled_arr3
    else:
        return scaled_arr1, scaled_arr2


def train_valid_test_split(X, y, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1, random_state=0):
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=1-train_ratio, random_state=random_state)
    xval, xtest, yval, ytest = train_test_split(
        xtest, ytest, test_size=test_ratio/(test_ratio+validation_ratio), random_state=random_state)

    return xtrain, xval, xtest, ytrain, yval, ytest


def create_submission(ypreds):

    # Get observation names
    files = os.listdir('../data/raw/test_data')
    npz_files = [os.path.splitext(file)[0]
                 for file in files if file.endswith('.npz')]

    submission = pd.DataFrame({
        'sample_index': npz_files,
        "P": ypreds[:, 0],
        "K": ypreds[:, 1],
        "Mg": ypreds[:, 2],
        "pH": ypreds[:, 3]
    })

    # Pivot to correct format
    melted_df = pd.melt(submission, id_vars=[
                        'sample_index'], var_name='var', value_name='Target')
    melted_df['sample_index'] = melted_df['sample_index'] + \
        '_' + melted_df['var']
    melted_df = melted_df[['sample_index', 'Target']]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f'../outputs/{timestamp}.csv'
    melted_df.to_csv(file_path, index=False)


def get_embeddings(X, newdata1=None, newdata2=None):

    scaler = StandardScaler()
    X = scaler.fit_transform(X=X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X=X)

    umap = UMAP(n_neighbors=15, n_components=3, random_state=0, n_jobs=1)
    X_umap = umap.fit_transform(X=X)

    tsne = TSNE(n_components=3, random_state=0)
    X_tsne = tsne.fit(X=X)

    X_embedded = np.hstack((X_pca, X_umap, X_tsne))

    if newdata1 is not None:
        newdata1 = scaler.transform(newdata1)
        newdata1_pca = pca.transform(newdata1)
        newdata1_umap = umap.transform(newdata1)
        newdata1_tsne = X_tsne.transform(newdata1)
        newdata_embedded1 = np.hstack(
            (newdata1_pca, newdata1_umap, newdata1_tsne))
    else:
        newdata_embedded1 = None

    if newdata2 is not None:
        newdata2 = scaler.transform(newdata2)
        newdata2_pca = pca.transform(newdata2)
        newdata2_umap = umap.transform(newdata2)
        newdata2_tsne = tsne.transform(newdata2)
        newdata_embedded2 = np.hstack(
            (newdata2_pca, newdata2_umap, newdata2_tsne))
    else:
        newdata_embedded2 = None

    if newdata1 is None and newdata2 is None:
        return X_embedded

    elif newdata2 is None and newdata1 is not None:
        return X_embedded, newdata_embedded1

    else:
        return X_embedded, newdata_embedded1, newdata_embedded2
