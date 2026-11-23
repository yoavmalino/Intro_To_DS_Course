import clustering
import numpy as np
from data import load_data, add_new_columns, data_analysis
def main():
    path = "london.csv"
    df = load_data(path)
    df = add_new_columns(df)
    print("Part A: ")
    data_analysis(df)

    print()
    print("Part B: ")
    k_options = [2, 3, 5]
    data = clustering.transform_data(df, ['cnt', 't1'])
    for k in k_options:
        labels, centroids = clustering.kmeans(data, k)
        print(f"k = {k}")
        print(np.array_str(centroids, precision=3, suppress_small=True))
        clustering.visualize_results(data, labels, centroids, f"k means {k}.pdf")
        if k != 5:
            print()

if __name__ == "__main__":
    main()