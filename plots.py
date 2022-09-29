import os
import matplotlib.pyplot as plt
import time


def save_losses(loss_list):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    import matplotlib

    matplotlib.use("Agg")
    plt.plot(len(loss_list), loss_list)
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(os.path.join("plots", timestr + ".png"))
