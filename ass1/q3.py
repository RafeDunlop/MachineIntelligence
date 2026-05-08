from q2 import all_possible_functions

def version_space(H, D):
    V = set()
    for h in H:
        if all([h(d[0]) == d[1] for d in D]):
            V.add(h)
    return V

if __name__ == "__main__":
    X = {"green", "purple"}  # an input space with two elements
    D = {("green", True)}  # the training data is a subset of X * {True, False}
    F = all_possible_functions(X)
    H = F  # H must be a subset of (or equal to) F

    VS = version_space(H, D)

    print(len(VS))

    for h in VS:
        for x, y in D:
            if h(x) != y:
                print("You have a hypothesis in VS that does not agree with the set D!")
                break
        else:
            continue
        break
    else:
        print("OK")