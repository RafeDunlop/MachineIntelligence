from q6 import zero_one_loss

def erm(D, H, loss):
    return min(H, key=lambda h: sum([loss(h(x), y) for x, y in D]))

if __name__ == '__main__':
    def h_always_true(x):
        return True


    def h_always_false(x):
        return False


    def h_is_even(x):
        return x % 2 == 0


    D = [(0, True), (1, False), (2, True), (3, False)]
    H = [h_always_true, h_always_false, h_is_even]
    print(erm(D, H, zero_one_loss) == h_is_even)