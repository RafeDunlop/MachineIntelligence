import sys

from helpers import partition_by_feature_value

def test1():
    from pprint import pprint

    dataset = [
      ((True, True), False),
      ((True, False), True),
      ((False, True), True),
      ((False, False), False),
    ]
    p = partition_by_feature_value(dataset, 0)
    pprint({k: sorted(v) for k, v in p.items()})

    partition_key = (True, True)[0]
    print(all(x[0] == True for x, c in p[partition_key]))
    partition_key = (False, True)[0]
    print(all(x[0] == False for x, c in p[partition_key]))

def test2():
    from pprint import pprint

    dataset = [
        (("a", "x", 2), False),
        (("b", "x", 2), False),
        (("a", "y", 5), True),
        (("c", "y", 5), False),
    ]
    p = partition_by_feature_value(dataset, 1)
    pprint({k: sorted(v) for k, v in p.items()})

    partition_key = ("b", "y", 5)[1]
    print(all(x[1] == "y" for x, c in p[partition_key]))


def test3():
    dataset = [
        (("Sunny", "Hot", "High", "Weak"), False),
        (("Sunny", "Hot", "High", "Strong"), False),
        (("Overcast", "Hot", "High", "Weak"), True),
        (("Rain", "Mild", "High", "Weak"), True),
        (("Rain", "Cool", "Normal", "Weak"), True),
        (("Rain", "Cool", "Normal", "Strong"), False),
        (("Overcast", "Cool", "Normal", "Strong"), True),
        (("Sunny", "Mild", "High", "Weak"), False),
        (("Sunny", "Cool", "Normal", "Weak"), True),
        (("Rain", "Mild", "Normal", "Weak"), True),
        (("Sunny", "Mild", "Normal", "Strong"), True),
        (("Overcast", "Mild", "High", "Strong"), True),
        (("Overcast", "Hot", "Normal", "Weak"), True),
        (("Rain", "Mild", "High", "Strong"), False),
    ]

    for feature_index in range(4):
        print(f"{feature_index}:", end=" ")
        partitions = partition_by_feature_value(dataset, feature_index)
        for value, partition in partitions.items():
            if not all(x_y[0][feature_index] == value for x_y in partition):
                print("Found incompatible values!")
                break
        else:
            print("OK")

if __name__ == "__main__":
    test3()