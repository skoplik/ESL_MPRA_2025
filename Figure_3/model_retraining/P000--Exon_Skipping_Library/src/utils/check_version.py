

def check_version(vA, vB):
    """
    Checks if vA is 'less', 'greater' or 
    'equal' to vB. vA and vB are period 
    separated strings denoting module
    versions.

    returns 'less' if vA is an older version.
    returns 'greater' if vA is a newer version.
    returns 'equal' if they are the same version.

    e.g. 2.2 vs 2.16 is less,
    but 2.2 < 2.16 is False alphabetically
    """
    vA = vA.split(".")
    vB = vB.split(".")
    a = 0
    b = 0
    vA_val = vA[a]
    vB_val = vB[b]
    while int(vA_val) == int(vB_val):
        a += 1
        b += 1
        if a > len(vA) - 1:
            vA_val = 0
        else:
            vA_val = vA[a]
        if b > len(vB) - 1:
            vB_val = 0
        else:
            vB_val = vB[b]
        if a > len(vA) - 1 and b > len(vB) - 1:
            break
    if int(vA_val) < int(vB_val):
        return "less"
    elif int(vA_val) > int(vB_val):
        return "greater"
    else:
        return "equal"


if __name__ == "__main__":
    import numpy as np
    tests = [
    (
     ".".join([
    np.random.choice([str(i) for i in range(20)]),\
    np.random.choice([str(i) for i in range(20)]),\
    np.random.choice([str(i) for i in range(20)]),\
    ]),

    ".".join([
    np.random.choice([str(i) for i in range(20)]),\
    np.random.choice([str(i) for i in range(20)]),\
    np.random.choice([str(i) for i in range(20)]),\
    ]),
    )
    for k in range(10)
    ]

    for i in tests:
        print(i[0], i[1], check_version(i[0], i[1]))
