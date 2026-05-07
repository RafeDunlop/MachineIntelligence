from itertools import permutations

def all_possible_functions(X):
    mag = len(X)
    X_list = list(X)
    F = set()
    classification_cardinalities = [[j < i for j in range(mag)] for i in range(mag+1)]
    for classification in classification_cardinalities:
        print(classification)
        F = F | set(classification)
        mappings = [{perm[i]: classification[i] for i in range(mag)} for perm in permutations(classification_cardinalities)]
        F |= set(mappings)
    return F

if __name__ == "__main__":
    X = {"green", "purple"}  # an input space with two elements
    F = all_possible_functions(X)

    # Let's store the image of each function in F as a tuple
    images = set()
    for h in F:
        images.add(tuple(h(x) for x in X))

    for image in sorted(images):
        print(image)