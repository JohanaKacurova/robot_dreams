from collections import deque
import numpy as np

def moving_average(xs, k=100):
    dq, out, s = deque(), [], 0.0
    for x in xs:
        dq.append(x)
        s += x
        if len(dq) > k:
            s -= dq.popleft()
        out.append(s / len(dq))
    return np.array(out)
