from itertools import product

def input_space(domains):
    return list(product(*domains))

if __name__ == '__main__':
    domains = [
        {0, 1, 2},
        {True, False},
    ]

    for element in sorted(input_space(domains)):
        print(element)
