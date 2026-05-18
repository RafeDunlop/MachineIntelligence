def autoencoder_widths(d, k, ratio):
    widths = []
    width = d
    while width > k:
        widths.append(width)
        width = int(width // ratio)
    return widths + [k] + list(reversed(widths))

def test1():
    print()
    widths = autoencoder_widths(80, 2, 2)
    print(type(widths))
    print(all(type(w) is int for w in widths))
    print(widths)

def test2():
    print()
    widths = autoencoder_widths(100, 5, 2.8)
    print(type(widths))
    print(all(type(w) is int for w in widths))
    print(widths)

if __name__ == "__main__":
    test1()
    test2()