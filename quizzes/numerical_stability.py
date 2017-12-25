"""Numerical Stability."""
import numpy as np
import matplotlib.pyplot as plt

billion = 1e+9
to_add = 1e-6
million = 1e+6

result = billion
for k in range(int(million)):
    result += to_add
    
final = result - billion
print("Final = {} = {} - {}".format(final, result, billion))
# The final result is not equal to 1.0, which was the theoretical result

# If the same test is made with 1 as initial value, we get a tiny error
one = 1

result = one
for k in range(int(million)):
    result += to_add
    
final = result - one
print("Final = {} = {} - {}".format(final, result, billion))