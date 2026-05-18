import numpy as np

def distance(feature_vector, centroid):
    return np.sum((feature_vector - centroid) ** 2)

def centroid_of(dataset):
    return np.mean(dataset, axis=0)

def bucket(dataset, centroids):
    buckets = [[] for _ in range(len(centroids))]
    closest = []
    for i in range(dataset.shape[0]):
        arg_min = min(range(len(centroids)), key=lambda j: distance(dataset[i, :], centroids[j]))
        closest.append(arg_min)
        buckets[arg_min].append(dataset[i, :])
    return buckets, closest

def k_means(dataset, centroids):
    assignment = [-1 for _ in range(len(dataset))]
    assignment_changed = True

    while assignment_changed:
        buckets, closest = bucket(dataset, centroids)
        assignment_changed = assignment != closest
        assignment = closest
        if assignment_changed:
            centroids = [centroid_of(buckets[j]) for j in range(len(centroids))]
    return centroids

def test1():
    dataset = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.8, 0.8],
        [0.9, 0.9]
    ])
    centroids = (np.array([0., 0.]), np.array([1., 1.]))
    for c in k_means(dataset, centroids):
        print(c)

def test2():
    print()
    dataset = np.array([
        [0.125, 0.125],
        [0.25, 0.25],
        [0.75, 0.75],
        [0.875, 0.875],
        [2.000, -1.00]
    ])
    centroids = (np.array([0., 1.]), np.array([1., 0.]))
    for c in k_means(dataset, centroids):
        print(c)

def test3():
    print()
    dataset = np.array([
        [0.1, 0.3],
        [0.4, 0.6],
        [0.1, 0.2],
        [0.2, 0.1]
    ])
    centroids = (np.array([2., 5.]),)
    for c in k_means(dataset, centroids):
        print(c)

def test4():
    print()
    import sklearn.datasets
    import sklearn.utils

    wine = sklearn.datasets.load_wine()
    data, target = sklearn.utils.shuffle(wine.data, wine.target, random_state=0)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]

    centroids = (
        np.array([13.0, 2.2, 2.4, 18.1, 107.9, 2.6,
                  2.5, 0.3, 1.6, 5.2, 1.0, 3.0, 964.0]),
        np.array([14.5, 1.8, 2.5, 17.0, 106.0, 2.9,
                  3.0, 0.3, 2.0, 6.6, 1.1, 3.0, 1300.0]),
        np.array([12.0, 3.1, 2.3, 20.7, 92.8, 2.0,
                  1.6, 0.4, 1.0, 4.7, 0.9, 2.0, 550.9])
    )
    for c in k_means(train_data, centroids):
        print(c)


if __name__ == '__main__':
    test4()



