from q3 import likelihood

def most_likely(samples, distributions):
    return max(distributions, key=lambda distribution: likelihood(samples, *distribution))

def test1():
    samples = [0.1]
    distributions = [(0, 1), (-2, 3)]
    mu, sigma = most_likely(samples, distributions)
    print(f"Sample most likely has mean {mu} and standard deviation {sigma}")

def test2():
    samples = [0.5]
    distributions = [(0, 1), (0, 0.5)]
    mu, sigma = most_likely(samples, distributions)
    print(f"Sample most likely has mean {mu} and standard deviation {sigma}")

if __name__ == "__main__":
    test1()
    test2()