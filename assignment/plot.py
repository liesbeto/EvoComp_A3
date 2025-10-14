import matplotlib.pyplot as plt
import pandas as pd


def make_plot(plot_filename, plot_y, save_path=None):
    plt.figure(figsize=(8, 4))
    df = pd.read_csv(plot_filename)
    y_data = df[plot_y]
    gens = [x for x in range(len(y_data))]
    plt.plot(gens, y_data, marker='o', markersize=3)
    plt.xlabel("Brain generation")
    plt.ylabel("Max fitness")
    plt.ylim(0, 1.5)
    plt.title("Maximum Fitness per Brain Generation")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # close figure
    else:
        plt.show()


def make_plot_two_y(plot1_filename, plot2_filename, plot_y, save_path=None):
    plt.figure(figsize=(8, 4))
    df1 = pd.read_csv(plot1_filename)
    df2 = pd.read_csv(plot2_filename)
    y_data1 = df1[plot_y]
    y_data2 = df2[plot_y]
    gens = [x for x in range(len(y_data1))]
    plt.plot(gens, y_data1, marker='o', markersize=5, label="EA")
    plt.plot(gens, y_data2, marker='o', markersize=5, label="Baseline")
    plt.xlabel("Body generation")
    plt.ylabel("Max fitness")
    plt.ylim(-0.1, 1.5)
    plt.legend()
    plt.title("Maximum Fitness per Generation")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close() # close figure
    else:
        plt.show()
