import numpy as np

def foo(x, y):
  x = np.array([x])
  y = np.array([y])
  return np.arctan2(y, x)
