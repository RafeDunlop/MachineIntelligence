def decode(code):
    x1, y1, x2, y2 = code
    return lambda z: in_range(z[0], x1, x2) and in_range(z[1], y1, y2)

def in_range(z, z1, z2):
    if z1 > z2:
        return in_range(z, z2, z1)
    return z1 <= z <= z2

if __name__ == '__main__':
    import itertools

    h = decode((-1, -1, 1, 1))

    for x in itertools.product(range(-2, 3), repeat=2):
        print(x, h(x))
    x