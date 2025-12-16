from ir.preprocess import preprocess

if __name__ == '__main__':
    test_str = "There are 3 balls in this bag, and 12 in the other one."
    print(preprocess(test_str))