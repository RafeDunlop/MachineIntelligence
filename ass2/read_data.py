def get_data(filename):
    dataset = []
    with open(filename + '.data', 'r') as f:
        for line in f.readlines():
            out, *features = line.strip().split(",")
            dataset.append((tuple(features), out))
    return dataset