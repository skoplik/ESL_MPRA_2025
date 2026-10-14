import os

def ignore_1(val):
    """
    for indexing starting at 1.
    e.g. run, run2, run3, ...
    """
    if val == 1 or val == 0:
        return ""
    else:
        return str(val)    


def get_new_dir(path_, suffix = "", debug = False):
    """
    Binary search to find the largest available run by checking 
    if the run directory exists at the given path.
    """
    path_num = []
    # get the model number given
    # TODO: technically path dirs could have . in them,
    # so should use os.path.split
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
    if not path_num:
        path_num = ["1"]
    path_num = "".join(path_num[::-1])
    path_num = int(path_num)
    path_ = path_[:i+1]
    if not os.path.exists(path_ + ignore_1(path_num) + suffix):
        if debug:
            breakpoint()
            print("returning condition 1")
        return path_ + ignore_1(path_num) + suffix, max(path_num, 1)
    path_num = max(path_num, 2)
    if not os.path.exists(path_ + suffix):
        if debug:
            print("returning condition 2")
        return path_, path_num

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
    if debug:
        print("returning condition 3")
    return path_ + ignore_1(l) + suffix, max(l, 1) # path, path_num


if __name__ == "__main__":
    print("test1")
    print(get_new_dir("test1"))
    print("test0")
    print(get_new_dir("test0"))
    print("test10")
    print(get_new_dir("test10"))
    print("test.hdf5")
    print(get_new_dir("test.hdf5", suffix = ".hdf5"))
    print("test0.hdf5")
    print(get_new_dir("test0.hdf5", suffix = ".hdf5"))
    print("test10.hdf5")
    print(get_new_dir("test10.hdf5", suffix = ".hdf5"))
    print("path_test1")
    print(get_new_dir("path_test1", debug = True))
    print("path_test0")
    print(get_new_dir("path_test0"))
    print("path_test10")
    print(get_new_dir("path_test10"))
    print("path_test1.hdf5")
    print(get_new_dir("path_test1.hdf5", suffix = ".hdf5"))
    print("path_test0.hdf5")
    print(get_new_dir("path_test0.hdf5", suffix = ".hdf5"))
    print("path_test10.hdf5")
    print(get_new_dir("path_test10.hdf5", suffix = ".hdf5"))
