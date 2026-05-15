import pandas as pd
from sklearn.utils import shuffle
import os


def main():
    os.makedirs("data", exist_ok=True)

    data = pd.read_csv("data/diabetes.csv")
    data = shuffle(data, random_state=42).reset_index(drop=True)

    total_rows = len(data)

    hospital1 = data.iloc[: total_rows // 3]
    hospital2 = data.iloc[total_rows // 3 : 2 * total_rows // 3]
    hospital3 = data.iloc[2 * total_rows // 3 :]

    hospital1.to_csv("data/hospital1.csv", index=False)
    hospital2.to_csv("data/hospital2.csv", index=False)
    hospital3.to_csv("data/hospital3.csv", index=False)

    print("Dataset successfully split into 3 hospitals")


if __name__ == "__main__":
    main()