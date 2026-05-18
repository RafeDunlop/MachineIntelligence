import random, collections, numpy as np
from q1 import k_means, distance, bucket

def cluster_distance(cluster1, cluster2):
    return min(distance(cluster1[i], cluster2[j]) for i in range(len(cluster1)) for j in range(len(cluster2)))

def separation(clusters):
    pairwise = []
    for i in range(len(clusters)):
        for j in range(i):
            pairwise.append(cluster_distance(clusters[i], clusters[j]))
    return sum(pairwise) / len(pairwise)

def diameter(cluster):
    return sum(
        distance(cluster[i], cluster[j])
        for i in range(len(cluster))
        for j in range(i)
    )

def compactness(clusters):
    return sum(
        diameter(cluster)
        for cluster in clusters
    ) / len(clusters)

def goodness(centroids, dataset):
    clusters, _ = bucket(dataset, centroids)
    return separation(clusters) / compactness(clusters)

def k_means_random_restart(dataset, k, restarts, seed=None):
    random.seed(seed)
    Model = collections.namedtuple('Model', 'goodness, centroids')
    models = []
    for _ in range(restarts):
        centroids = k_means(dataset, random.sample([x for x in dataset], k=k))
        models.append(Model(goodness(centroids, dataset), centroids))
    return max(models, key=lambda m: m.goodness).centroids

def test1():
    print()
    dataset = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.8, 0.8],
        [0.9, 0.9]
    ])
    centroids = k_means_random_restart(dataset, k=2, restarts=5, seed=0)

    for c in sorted([f"{x:.3}" for x in centroid] for centroid in centroids):
        print("  ".join(c))

def test2():
    print()
    import sklearn.datasets
    import sklearn.utils

    iris = sklearn.datasets.load_iris()
    data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]

    centroids = k_means_random_restart(iris.data, k=3, restarts=10, seed=0)

    # We suggest you check which centroid each
    # element in test_data is closest to, then see test_target.
    # Note cluster 0 -> label 1
    #      cluster 1 -> label 2
    #      cluster 2 -> label 0

    for c in sorted([f"{x:.2}" for x in centroid] for centroid in centroids):
        print("  ".join(c))

def test3():
    print()
    import sklearn.datasets

    wine = sklearn.datasets.load_wine()
    centroids = k_means_random_restart(wine.data, k=3, restarts=10, seed=0)
    for c in sorted([f"{x:.1f}" for x in centroid] for centroid in centroids):
        print("  ".join(c))

if __name__ == "__main__":
    test3()