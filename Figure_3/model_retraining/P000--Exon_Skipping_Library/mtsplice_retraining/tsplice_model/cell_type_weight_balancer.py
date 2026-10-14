import numpy as np



def get_cell_type_weights(y_train, method):
    """

    Input: numpy array (labels) for training dataset
           method: if it is a list of floats, return those as the weights
                   otherwise, if 'proportional', set the weights to 
                   be proportional to the number of NaNs. If 'inverse',
                   set the weights to be inversely proportional to the
                   number of NaNs. If 'balanced', set all the weights to
                   the default 1.0 for all cell types
    Output: cell type weights, proportional to the fraction of NaNs 
    """
    if isinstance(method, list):
        # hardcoded weights passed
        return np.array(method)
    
    match method:
        case "proportional":
            reverse = False
        case "inverse":
            reverse = True 
        case "balanced":
            return np.ones((5,))
        case _:
            raise ValueError(
                f"method '{method}' not known "
                f"for getting cell weights. "
                f"must be 'proportional', 'inverse' or 'balanced'"
            )

    is_nan = np.isnan(y_train)
    nan_per_cell = np.sum(is_nan, axis = 0)
    total = np.sum(nan_per_cell)
    if total == 0:
        return np.array([1.0] * len(nan_per_cell))
    if reverse:
        total = np.sum(1 / nan_per_cell)
        nan_per_cell = 1 / nan_per_cell
    cell_type_weights = nan_per_cell / total * len(nan_per_cell)
    return cell_type_weights


def score_based_cell_weights(scores, scale=5):
    """ 
    Scale weights according to scores such
    that the total sum is equal to scale.
    """
    proportions = [i / np.sum(scores) for i in scores]
    nproportions = np.array(proportions)
    inv = 1 / nproportions
    return (inv * scale) / np.sum(inv)


def parse_cell_type_weights(string):
    """
    Split miltiple cell weights
    (i.e. for different heads)
    into a list of lists, where
    each list contains weights
    for all 5 cell types.
    """
    if ";" not in string:
        return parse_weight(string)
    string = string.split(";")
    return [parse_weight(i) for i in string]


def parse_weight(string):
    """
    Split comma-separated cell weights.
    """
    if not string:
        print(
            "Warning: cell type weight not found, "
            "using 1.0 for all cells"
        )
        return "balanced"
    if "," in string:
        string = string.split(",")
        return [float(i) for i in string]
    return string


if __name__ == "__main__":
    print("testing cell type balancer")
    y_train = np.array([
        [1,2,3],
        [np.nan, np.nan, 4],
        [5, np.nan, 6],
        [4,4,np.nan],
        [6,7,np.nan],
        [6,9,np.nan]
    ])
    
    wts=get_cell_type_weights(y_train, "proportional")
    print(wts)
