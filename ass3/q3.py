from operator import mul
import numpy as np
from functools import reduce

def conditional_density_normal(sample, mu, sigma):
    normalising_coefficient = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    return normalising_coefficient * np.exp(- (sample - mu) ** 2 / (2 * sigma ** 2))

def likelihood(samples, mu, sigma):
    sample_likelihoods = map(lambda sample: conditional_density_normal(sample, mu, sigma), samples)
    # failed before because i renamed mul to product and a test used `itertools.product` 🤦
    return reduce(mul, sample_likelihoods)

def test1():
    mu = 0
    sigma = 1
    samples = [0.2]
    print(f"{likelihood(samples, mu, sigma):.4f}")

def test2():
    mu = 0
    sigma = 1
    samples = [-2.2]
    print(f"{likelihood(samples, mu, sigma):.4f}")

if __name__ == "__main__":
    test1()
    test2()


