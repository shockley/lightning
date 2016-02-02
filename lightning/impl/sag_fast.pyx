# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG

from libc.math cimport sqrt

from lightning.impl.randomkit.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport RowDataset
from lightning.impl.sgd_fast cimport LossFunction


cdef double _pred(double* data,
                  int* indices,
                  int n_nz,
                  double* w):

    cdef int j, jj
    cdef double dot = 0

    for jj in xrange(n_nz):
        j = indices[jj]
        dot += w[j] * data[jj]

    return dot


cdef void _add(double* data,
               int* indices,
               int n_nz,
               double scale,
               double* w):
    cdef int jj, j

    for jj in xrange(n_nz):
        j = indices[jj]
        w[j] += scale * data[jj]


#shockley
def _total_loss(self,
             np.ndarray[double, ndim=1, mode='c'] w,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             int loss):
    if loss != 2:
        #only supports hinge, more losses will be supported later
        return -1.0
    cdef double LOSS = 0.0
    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()
    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for i in xrange(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        
        #get prediction
        p = 0.0
        for jj in xrange(n_nz):
            j = indices[jj]
            p += w[j] * data[jj]
        
        #get loss (hinge)
        z = p * y[i]
        if z <= 1.0:
            LOSS += (1.0 - z)
        #LOSS += 0.0
    return LOSS

def _sag_fit(self,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             np.ndarray[double, ndim=1]coef,
             np.ndarray[double, ndim=1]coef_scale,
             np.ndarray[double, ndim=1]grad,
             double eta,
             double alpha,
             LossFunction loss,
             int max_iter,
             int n_inner,
             double tol,
             int verbose,
             callback,
             RandomState rng):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables.
    cdef int i, jj, j, it, t
    cdef double y_pred, scale, g_old, tmp, alpha_scaled
    cdef double violation, violation_init, violation_ratio
    cdef double eta_avg = eta / n_samples
    cdef double eta_alpha = eta * alpha
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef np.ndarray[int, ndim=1]last = np.zeros(n_features, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] g_sum_
    g_sum_ = np.zeros(n_features, dtype=np.float64)
    cdef double* g_sum = <double*>g_sum_.data
    cdef double* w = <double*>coef.data
    cdef double* w_scale = <double*>coef_scale.data
    cdef double* g = <double*>grad.data

    # Initialize gradient memory.
    for i in xrange(n_samples):
        # Retrieve sample i.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        # Make prediction.
        y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

        # A gradient is given by g[i] * X[i].
        g[i] = -loss.get_update(y_pred, y[i])

        # Update g_sum.
        _add(data, indices, n_nz, g[i], g_sum)

    # Outer loop.
    for it in xrange(max_iter):

        # Inner loop.
        for t in xrange(n_inner):
            i = rng.randint(n_samples - 1)

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Update coefficients, just in time.
            if t > 0:
                for jj in xrange(n_nz):
                    j = indices[jj]
                    w[j] -= eta_avg / w_scale[0] * (t - last[j]) * g_sum[j]
                    last[j] = t

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

            # Make copy of old gradient value.
            g_old = g[i]

            # A gradient is given by g[i] * X[i].
            g[i] = -loss.get_update(y_pred, y[i])

            # Update g_sum.
            _add(data, indices, n_nz, (g[i] - g_old), g_sum)

            # Update coefficient scale (l2 regularization).
            w_scale[0] *= (1 - eta_alpha)

            # Take care of possible underflows.
            if w_scale[0] < 1e-9:
                for j in xrange(n_features):
                    w[j] *= w_scale[0]
                w_scale[0] = 1.0

        # Finalize.
        for j in xrange(n_features):
            w[j] -= eta_avg / w_scale[0] * (n_inner - last[j]) * g_sum[j]
            last[j] = 0

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break

        # Compute optimality violation.
        violation = 0
        alpha_scaled = alpha * w_scale[0]
        for j in xrange(n_features):
            tmp = g_sum[j] / n_samples + alpha_scaled * w[j]
            violation += tmp * tmp
        violation = sqrt(violation)

        # Convergence monitoring.
        if it == 0:
            violation_init = violation

        violation_ratio = violation / violation_init

        if verbose:
            print it + 1, violation_ratio

        if violation_ratio <= tol:
            if verbose:
                print "Converged"
            break

    # Rescale coefficients.
    for j in xrange(n_features):
        w[j] *= w_scale[0]
    w_scale[0] = 1.0
