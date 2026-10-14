import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def save_splits(one_hot_splits_dict, data_dir):
    pickle.dump(one_hot_splits_dict["x_train"], open(os.path.join(data_dir, "x_train.p"), "wb"))
    pickle.dump(one_hot_splits_dict["x_val"],   open(os.path.join(data_dir, "x_val.p"), "wb"))
    pickle.dump(one_hot_splits_dict["x_test"],  open(os.path.join(data_dir, "x_test.p"), "wb"))
    pickle.dump(one_hot_splits_dict["y_train"], open(os.path.join(data_dir, "y_train.p"), "wb"))
    pickle.dump(one_hot_splits_dict["y_val"],   open(os.path.join(data_dir, "y_val.p"), "wb"))
    pickle.dump(one_hot_splits_dict["y_test"],  open(os.path.join(data_dir, "y_test.p"), "wb"))
    pd.DataFrame(one_hot_splits_dict["metadata_train"]).to_csv(os.path.join(data_dir, "metadata_train.csv"), index = False)
    pd.DataFrame(one_hot_splits_dict["metadata_val"]).to_csv(os.path.join(data_dir, "metadata_val.csv"), index = False)
    pd.DataFrame(one_hot_splits_dict["metadata_test"]).to_csv(os.path.join(data_dir, "metadata_test.csv"), index = False)
    return



def psi_distribution_plot(train, val, test, data_dir, scale = "PSI"):
    # PSI distributions of train, val and test
    fig, ax = plt.subplots(3, figsize = (15, 20))
    ax[0].hist([train[i][3] for i in range(len(train))])
    ax[0].set_title("Training Set", fontsize = 15)
    ax[1].hist([val[i][3] for i in range(len(val))])
    ax[1].set_title("Validation Set", fontsize = 15)
    ax[2].hist([test[i][3] for i in range(len(test))])
    ax[2].set_title("Test Set", fontsize = 15)
    fig.supxlabel("PSI", fontsize = 20)
    fig.supylabel("Frequency", fontsize = 20)
    fig.suptitle(f"Distribution of {scale} Across Data Splits", fontsize = 25)
    save_dist_name = scale.lower().replace(" ", "_")
    plt.savefig(os.path.join(data_dir, f"{save_dist_name}_distribution.pdf"))
    plt.savefig(os.path.join(data_dir, f"{save_dist_name}_distribution.png"))
    return
