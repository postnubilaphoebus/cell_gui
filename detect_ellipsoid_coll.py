import numpy as np
import numba
import time
from numba import njit

@numba.njit()
def accelerated_function(A, B, Ma, Mb):

    aux = np.linalg.solve(Mb, Ma)
    b = aux.T @ B @ aux

    # Coefficients of the Characteristic Polynomial
    T4 = (-A[0, 0] * A[1, 1] * A[2, 2])
    T3 = (A[0, 0] * A[1, 1] * b[2, 2] + A[0, 0] * A[2, 2] * b[1, 1] + A[1, 1] * A[2, 2] * b[0, 0] - A[0, 0] * A[1, 1] * A[2, 2] * b[3, 3])
    T2 = (A[0, 0] * b[1, 2] * b[2, 1] - A[0, 0] * b[1, 1] * b[2, 2] - A[1, 1] * b[0, 0] * b[2, 2] + A[1, 1] * b[0, 2] * b[2, 0] - 
          A[2, 2] * b[0, 0] * b[1, 1] + A[2, 2] * b[0, 1] * b[1, 0] + A[0, 0] * A[1, 1] * b[2, 2] * b[3, 3] - A[0, 0] * A[1, 1] * b[2, 3] * b[3, 2] + 
          A[0, 0] * A[2, 2] * b[1, 1] * b[3, 3] - A[0, 0] * A[2, 2] * b[1, 3] * b[3, 1] + A[1, 1] * A[2, 2] * b[0, 0] * b[3, 3] - 
          A[1, 1] * A[2, 2] * b[0, 3] * b[3, 0])
    T1 = (b[0, 0] * b[1, 1] * b[2, 2] - b[0, 0] * b[1, 2] * b[2, 1] - b[0, 1] * b[1, 0] * b[2, 2] + b[0, 1] * b[1, 2] * b[2, 0] + 
          b[0, 2] * b[1, 0] * b[2, 1] - b[0, 2] * b[1, 1] * b[2, 0] - A[0, 0] * b[1, 1] * b[2, 2] * b[3, 3] + A[0, 0] * b[1, 1] * b[2, 3] * b[3, 2] + 
          A[0, 0] * b[1, 2] * b[2, 1] * b[3, 3] - A[0, 0] * b[1, 2] * b[2, 3] * b[3, 1] - A[0, 0] * b[1, 3] * b[2, 1] * b[3, 2] + 
          A[0, 0] * b[1, 3] * b[2, 2] * b[3, 1] - A[1, 1] * b[0, 0] * b[2, 2] * b[3, 3] + A[1, 1] * b[0, 0] * b[2, 3] * b[3, 2] + 
          A[1, 1] * b[0, 2] * b[2, 0] * b[3, 3] - A[1, 1] * b[0, 2] * b[2, 3] * b[3, 0] - A[1, 1] * b[0, 3] * b[2, 0] * b[3, 2] + 
          A[1, 1] * b[0, 3] * b[2, 2] * b[3, 0] - A[2, 2] * b[0, 0] * b[1, 1] * b[3, 3] + A[2, 2] * b[0, 0] * b[1, 3] * b[3, 1] + 
          A[2, 2] * b[0, 1] * b[1, 0] * b[3, 3] - A[2, 2] * b[0, 1] * b[1, 3] * b[3, 0] - A[2, 2] * b[0, 3] * b[1, 0] * b[3, 1] + 
          A[2, 2] * b[0, 3] * b[1, 1] * b[3, 0])
    T0 = (b[0, 0] * b[1, 1] * b[2, 2] * b[3, 3] - b[0, 0] * b[1, 1] * b[2, 3] * b[3, 2] - b[0, 0] * b[1, 2] * b[2, 1] * b[3, 3] + 
          b[0, 0] * b[1, 2] * b[2, 3] * b[3, 1] + b[0, 0] * b[1, 3] * b[2, 1] * b[3, 2] - b[0, 0] * b[1, 3] * b[2, 2] * b[3, 1] - 
          b[0, 1] * b[1, 0] * b[2, 2] * b[3, 3] + b[0, 1] * b[1, 0] * b[2, 3] * b[3, 2] + b[0, 1] * b[1, 2] * b[2, 0] * b[3, 3] - 
          b[0, 1] * b[1, 2] * b[2, 3] * b[3, 0] - b[0, 1] * b[1, 3] * b[2, 0] * b[3, 2] + b[0, 1] * b[1, 3] * b[2, 2] * b[3, 0] + 
          b[0, 2] * b[1, 0] * b[2, 1] * b[3, 3] - b[0, 2] * b[1, 0] * b[2, 3] * b[3, 1] - b[0, 2] * b[1, 1] * b[2, 0] * b[3, 3] + 
          b[0, 2] * b[1, 1] * b[2, 3] * b[3, 0] + b[0, 2] * b[1, 3] * b[2, 0] * b[3, 1] - b[0, 2] * b[1, 3] * b[2, 1] * b[3, 0] - 
          b[0, 3] * b[1, 0] * b[2, 1] * b[3, 2] + b[0, 3] * b[1, 0] * b[2, 2] * b[3, 1] + b[0, 3] * b[1, 1] * b[2, 0] * b[3, 2] - 
          b[0, 3] * b[1, 1] * b[2, 2] * b[3, 0] - b[0, 3] * b[1, 2] * b[2, 0] * b[3, 1] + b[0, 3] * b[1, 2] * b[2, 1] * b[3, 0])

    # t4 = time.time()

    # total_time = t4 - t1
    # fraction_time1 = (t2 - t1) / total_time
    # fraction_time2 = (t3 - t2) / total_time
    # fraction_time3 = (t4 - t3) / total_time

    # print(f"Total time: {total_time}")
    # print(f"Fraction time 1: {fraction_time1}")
    # print(f"Fraction time 2: {fraction_time2}")
    # print(f"Fraction time 3: {fraction_time3}")

    # import pdb; pdb.set_trace()
    

    # Roots of the characteristic polynomial
    characteristic_polynomial = np.array([T4, T3, T2, T1, T0])
    

    return characteristic_polynomial
@njit
def add_extra_row_and_concat(A_i, A_j, r_i, r_j, extra_row):
    T_i = np.vstack((np.hstack((A_i, r_i.reshape(-1, 1))), extra_row))
    T_j = np.vstack((np.hstack((A_j, r_j.reshape(-1, 1))), extra_row))
    return T_i, T_j

import numpy.polynomial.polynomial as poly
from scipy.interpolate import PPoly

@njit
def find_roots(characteristic_polynomial):
    # import time
    # t0 = time.time()
    r = np.roots(characteristic_polynomial.astype(np.complex128))
    # r = poly.polyroots(characteristic_polynomial.astype(np.complex128))
    #t1 = time.time()
    r[np.abs(r.imag) <= 1e-3] = r[np.abs(r.imag) <= 1e-3].real
    #t2 = time.time()
    complex_roots = np.iscomplex(r)
    #t3 = time.time()
    negative_roots_ids = np.where((~complex_roots) & (r.real < 0))[0]
    # t4 = time.time()
    # total_time = t4 - t0
    # fraction_time1 = (t1 - t0) / total_time
    # fraction_time2 = (t2 - t1) / total_time
    # fraction_time3 = (t3 - t2) / total_time
    # fraction_time4 = (t4 - t3) / total_time

    # print(f"Total time: {total_time}")
    # print(f"Fraction time 1: {fraction_time1}")
    # print(f"Fraction time 2: {fraction_time2}")
    # print(f"Fraction time 3: {fraction_time3}")
    # print(f"Fraction time 4: {fraction_time4}")
    #import pdb; pdb.set_trace()
    return negative_roots_ids, r

# @njit
def algebraic_separation_condition(coeff_canon_i, coeff_canon_j, r_i, r_j, A_i, A_j, extra_row):
    """
    Algebraic condition stating the contact status between two ellipsoids.
    
    Parameters:
    coeff_canon_i : array-like
        Ellipsoid radii (x, y, z) of surface i.
    coeff_canon_j : array-like
        Ellipsoid radii (x, y, z) of surface j.
    r_i : array-like
        Position vector of surface i's centroid.
    r_j : array-like
        Position vector of surface j's centroid.
    A_i : array-like
        Rotation matrix of surface i.
    A_j : array-like
        Rotation matrix of surface j.
    
    Returns:
    status : str
        'y' if ellipsoids are apart and 'n' if overlapped.
    """
    #import time
    A = np.array([
        [1/coeff_canon_i[0]**2, 0, 0, 0],
        [0, 1/coeff_canon_i[1]**2, 0, 0],
        [0, 0, 1/coeff_canon_i[2]**2, 0],
        [0, 0, 0, -1]
    ])

    B = np.array([
        [1/coeff_canon_j[0]**2, 0, 0, 0],
        [0, 1/coeff_canon_j[1]**2, 0, 0],
        [0, 0, 1/coeff_canon_j[2]**2, 0],
        [0, 0, 0, -1]
    ])
    #t0 = time.time()
    T_i, T_j = add_extra_row_and_concat(A_i, A_j, r_i, r_j, extra_row)
    #t1 = time.time()
    characteristic_polynomial = accelerated_function(A, B, T_i, T_j)
    #t2 = time.time()
    negative_roots_ids, r = find_roots(characteristic_polynomial)
    #t3 = time.time()
    # total_time = t3 - t0
    # fraction_time1 = (t1 - t0) / total_time
    # fraction_time2 = (t2 - t1) / total_time
    # fraction_time3 = (t3 - t2) / total_time

    # print(f"Total time: {total_time}")
    # print(f"Fraction time 1: {fraction_time1}")
    # print(f"Fraction time 2: {fraction_time2}")
    # print(f"Fraction time 3: {fraction_time3}")

    # import pdb; pdb.set_trace()
    # Contact detection status
    if len(negative_roots_ids) == 2:
        if r[negative_roots_ids[0]] != r[negative_roots_ids[1]]:
            return 0
        elif np.abs(r[negative_roots_ids[0]] - r[negative_roots_ids[1]]) <= 1e-3:
            return 1
    else:
        return 1