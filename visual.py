import json
import argparse

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    n_range = 10

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="Dirpath for visualisation")

    args = parser.parse_args()
    path = args.path

    with open(f"{path}/results.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    _, canvas = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(25, 12)
    )
    
    for i, group in enumerate(data.keys()):
        for j, metric in enumerate(data[group].keys()):
            values = data[group][metric]

            canvas[i][j].set_xticks(np.arange(0, len(values) + 1, n_range))
            canvas[i][j].grid(axis='x', linestyle='-', linewidth=2, color='gray')
            canvas[i][j].grid(axis='y', linestyle='--', alpha=0.5)

            canvas[i][j].plot(list(range(len(values))), values, "g-o")
            canvas[i][j].set_title(f"{group}/{metric}")
    
    plt.savefig(f"{path}/results.png", format="png", dpi=600)
    # plt.show()
