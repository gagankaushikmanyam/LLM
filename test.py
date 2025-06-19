import json
import numpy as np
import matplotlib.pyplot as plt
import os

import time

# IMPORTING SETTINGS TO SAVE DATA, FIGURE
import settings

np.random.seed(42)

FIGURES_PATH = os.path.join(
        settings.get_output_path("Figures/test/"),
        
    )

# Ensure the directory exists
if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)

if __name__ == "__main__":

    ############################################## STEP 0: EXTRACTING INPUTS ##############################################

    with open("input_data.json", "r") as file:
        input_data = json.load(file)

    test_array_length = input_data["test_array_length"]
    print(test_array_length)

    test_array = np.ones(test_array_length)
    print(test_array)

    test_array_2 = 10* np.ones(test_array_length)
    print(test_array_2)

    plt.figure(0)
    plt.plot(test_array)
    plt.savefig(FIGURES_PATH + "test.png")

    plt.figure(1)
    plt.plot(test_array_2)
    plt.savefig(FIGURES_PATH + "test2.png")