import math

def mean(samples):
    return sum(samples) / len(samples)

def standard_deviation(samples, mu):
    return math.sqrt(sum(map(lambda sample: (sample - mu) ** 2, samples)) / len(samples))

def max_log_likelihood_estimator(samples):
    mu = mean(samples)
    return mu, standard_deviation(samples, mu)

def test1():
    samples = [-0.5, 0.5]
    mu, sigma = max_log_likelihood_estimator(samples)
    print(mu == 0, sigma == 0.5)
    print(f"{mu:.4f}", f"{sigma:.4f}")

def test2():
    import numpy as np
    samples = np.full(100, -0.25)
    mu, sigma = max_log_likelihood_estimator(samples)
    print(mu == -0.25, sigma == 0)
    print(f"{mu:.4f}", f"{sigma:.4f}")

if __name__ == "__main__":
    test1()
    test2()
