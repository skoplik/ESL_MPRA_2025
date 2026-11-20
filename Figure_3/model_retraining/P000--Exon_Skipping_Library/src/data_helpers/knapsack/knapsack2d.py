import numpy as np


def knapsack2D(pool, max_sizeA, max_sizeB, A = 0, B = 4, score_index = 1):
        n = len(pool)
        m = np.zeros((n + 1, max_sizeA + 1, max_sizeB + 1))
        for i in range(1,n+1):
                for j in range(1,max_sizeA+1):
                        for k in range(1, max_sizeB+1):
                                if pool[i-1][A] > j or pool[i-1][B] > k:
                                        m[i][j][k] = m[i-1][j][k]
                                else:
                                        m[i][j][k] = max(m[i-1][j][k], m[i-1][j-pool[i-1][A]][k-pool[i-1][B]] + pool[i-1][score_index])
        return m



def traceback2D(m, pool, A = 0, B = 4):
	chosen = set()
	i, j, k = len(m) - 1, len(m[0]) - 1, len(m[0][0]) - 1
	while i > 0:
		initial = (i, j, k)
		if m[i - 1][j][k] == m[i][j][k]:
			i -= 1
		else:
			chosen.add(i-1)
			j = j - pool[i-1][A]
			k = k - pool[i-1][B]
			i = i - 1
	return chosen


