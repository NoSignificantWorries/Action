import json
import argparse

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    n_range = 5

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="Dirpath for visualisation")

    args = parser.parse_args()
    path = args.path

    with open(f"{path}/results.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    _, canvas = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(25, 12)
    )
    
    for i, metric in enumerate(data.keys()):
        values = data[metric]

        canvas[i // 2][i % 2].set_xticks(np.arange(0, len(values) + 1, n_range))
        canvas[i // 2][i % 2].grid(axis='x', linestyle='-', linewidth=2, color='gray')
        canvas[i // 2][i % 2].grid(axis='y', linestyle='--', alpha=0.5)

        canvas[i // 2][i % 2].plot(list(range(len(values))), values, "g-o")
        canvas[i // 2][i % 2].set_title(f"{metric}")
    
    plt.savefig(f"{path}/results.png", format="png", dpi=600)
    # plt.show()
