# Import required packages 
import umap
import umap.plot
import numpy as np
import pandas as pd
import numpy as np
import sklearn.datasets 
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', title='', show=False):
    """
    Params:
    - n_neighbors (int): 
        This parameter controls how UMAP balances local versus global structure in the data. 
        It does this by constraining the size of the local neighborhood UMAP will look at when
        attempting to learn the manifold structure of the data.

    - min_dist (float): 
        The min_dist parameter controls how tightly UMAP is allowed to pack points together. 
        It, quite literally, provides the minimum distance apart that points are allowed to be in 
        the low dimensional representation.

    - n_components (int):
        Determines the dimensionality of the embedding space

    - metric (str):
        This controls how distance is computed in the ambient space of the input data. 
        By default UMAP supports a wide variety of metrics, including:
        euclidean, manhattan, chebyshev, minkowski, cosine, correlation
    """
    mapper = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        n_jobs=28,
        verbose=True
    ).fit(data)

    embeddings = mapper.embedding_

    with open(f"/scratch/GIT/BikeML/embeddings/umap_viz/{title}.npz","wb") as f:
        np.save(f, embeddings)

    fig = umap.plot.points(mapper)
    plt.savefig(f'/scratch/GIT/BikeML/embeddings/umap_viz/n_neighbors_{n_neighbors}.pdf', format='pdf')
    if show:
        plt.show()

if __name__ == "__main__":
    name = 'big_ass_hist_embeddings.npz'
    data = np.load(name)
    for n in (5, 10, 25, 50, 100, 200, 500):
        draw_umap(data, n_neighbors=n, title=f'n_neighbors_{n}')

    # mnist = sklearn.datasets.fetch_openml('mnist_784')
    # # vectors = np.load('vectors.npz')
    # # print(vectors.shape)
    # mapper = umap.UMAP(random_state=42).fit(mnist.data)
    # umap.plot.points(mapper)
    # plt.show()