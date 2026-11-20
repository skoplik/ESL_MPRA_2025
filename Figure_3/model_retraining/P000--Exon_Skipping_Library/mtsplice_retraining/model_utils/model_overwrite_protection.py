import os

def ignore_1(val):
	# for indexing starting at 1.
	# e.g. run, run1, run2, ...
	if val == 1:
		return ""
	else:
		return str(val)	


def get_new_dir(path_, suffix = ""):
    """
    Binary search to find the largest available run by checking 
    if the run directory exists at the given path.
    """
    path_num = []
    # get the model number given
    # TODO: technically path dirs could have . in them, so should use os.path.split
    if "." in path_: # for files, e.g. model12.hdf5
        file_path = path_.split(".")
        file_name = file_path[0]
        extension = file_path[-1]
        assert suffix, "file extension not given but exists in input"
    else: # for directories, e.g. models/model10
        file_name = path_
        assert not suffix, "no file extension given but exists in given path"
    i = len(file_name) - 1
    while i >= 0 and file_name[i].isnumeric():
        path_num.append(file_name[i])
        i -= 1
    if not os.path.exists(path_):
        if not path_num:
            path_num = [1]
        return path_, path_num[0]
    if not path_num:
        path_num = 2
    else:
        path_num = "".join(path_num[::-1])
        path_num = int(path_num)
    if not os.path.exists(path_):
            return path_, path_num
    path_ = file_name
    path_ = path_[:i+1]
    while os.path.exists(path_ + ignore_1(path_num) + suffix):
        path_num *= 2
    l = path_num // 2
    r = path_num    
    while l <= r:
        m = (l + r) // 2
        exists = os.path.exists(path_ + ignore_1(m) + suffix)
        if exists:
            l = m + 1
        else:
            r = m - 1
    return path_ + ignore_1(l) + suffix, l # path, path_num

