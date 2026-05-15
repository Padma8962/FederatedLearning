from client.preprocess import load_data
from client.model import create_model


def main():
    X, y = load_data("diabetes.csv")

    model = create_model()
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    loss, acc = model.evaluate(X, y, verbose=0)
    print("Accuracy:", float(acc))


if __name__ == "__main__":
    main()