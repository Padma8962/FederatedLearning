import matplotlib.pyplot as plt


def main():
    rounds = [1, 2, 3, 4, 5]
    accuracy = [0.71, 0.75, 0.78, 0.80, 0.82]

    plt.figure()
    plt.plot(rounds, accuracy, marker="o")
    plt.xlabel("Federated Rounds")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Accuracy vs Rounds")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()