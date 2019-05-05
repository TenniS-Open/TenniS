import tensorstack as ts


def test():
    a = 1
    t = ts.tensor.from_any(a)

    print("Before t = {}".format(t))

    with open("test.t", "wb") as fo:
        ts.tensor.write_tensor(fo, t)

    with open("test.t", "rb") as fi:
        t = ts.tensor.read_tensor(fi)

    print("After 1 t = {}".format(t))

    with open("test.t", "wb") as fo:
        ts.tensor.write_tensor(fo, t)

    with open("test.t", "rb") as fi:
        t = ts.tensor.read_tensor(fi)

    print("After 2 t = {}".format(t))

    a = ts.tensor.PackedTensor((1, "asdf", [2, 3]))

    print("Before a = {}".format(repr(a)))

    ts.tensor.write("packed.t", a)

    a = ts.tensor.read("packed.t")

    print("After a = {}".format(repr(a)))


if __name__ == '__main__':
    test()