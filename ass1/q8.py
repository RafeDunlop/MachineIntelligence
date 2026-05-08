from math import exp

def probability_lower_bound(test_outcomes, deviation):
    return 1 - 2 * exp(-2 * len(test_outcomes) * deviation ** 2)

if __name__ == '__main__':
    print(probability_lower_bound([True, False] * 500, 0.05))