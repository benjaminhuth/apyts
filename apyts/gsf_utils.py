import logging

from scipy.stats import multivariate_normal
import numpy as np

def approx_bethe_heitler_distribution(x_in_x0, single_component_approx):
    if single_component_approx:
        c = x_in_x0 / np.log(2)
        return [(1.0, 2**(-c), 3**(-c) - 4**(-c))]

    kBetheHeitlerPolynomialApprox = [
        # Component #1
        (
            [3.74397e+004,-1.95241e+004, 3.51047e+003,-2.54377e+002, 1.81080e+001,-3.57643e+000],
            [3.56728e+004,-1.78603e+004, 2.81521e+003,-8.93555e+001,-1.14015e+001, 2.55769e-001],
            [3.73938e+004,-1.92800e+004, 3.21580e+003,-1.46203e+002,-5.65392e+000,-2.78008e+000]
        ),
        # Component #2
        (
            [-4.14035e+004, 2.31883e+004,-4.37145e+003, 2.44289e+002, 1.13098e+001,-3.21230e+000],
            [-2.06936e+003, 2.65334e+003,-1.01413e+003, 1.78338e+002,-1.85556e+001, 1.91430e+000],
            [-5.19068e+004, 2.55327e+004,-4.22147e+003, 1.90227e+002, 9.34602e+000,-4.80961e+000]
        ),
        # Component #3
        (
            [2.52200e+003,-4.86348e+003, 2.11942e+003,-3.84534e+002, 2.94503e+001,-2.83310e+000],
            [1.80405e+003,-1.93347e+003, 6.27196e+002,-4.32429e+001,-1.43533e+001, 3.58782e+000],
            [-4.61617e+004, 1.78221e+004,-1.95746e+003,-8.80646e+001, 3.43153e+001,-7.57830e+000]
        ),
        # Component #4
        (
            [4.94537e+003,-2.08737e+003, 1.78089e+002, 2.29879e+001,-5.52783e+000,-1.86800e+000],
            [4.60220e+003,-1.62269e+003,-1.57552e+002, 2.01796e+002,-5.01636e+001, 6.47438e+000],
            [-9.50373e+004, 4.05517e+004,-5.62596e+003, 4.58534e+001, 6.70479e+001,-1.22430e+001]
        ),
        # Component #5
        (
            [-1.04129e+003, 1.15222e+002,-2.70356e+001, 3.18611e+001,-7.78800e+000,-1.50242e+000],
            [-2.71361e+004, 2.00625e+004,-6.19444e+003, 1.10061e+003,-1.29354e+002, 1.08289e+001],
            [3.15252e+004,-3.31508e+004, 1.20371e+004,-2.23822e+003, 2.44396e+002,-2.09130e+001]
        ),
        # Component #6
        (
            [1.27751e+004,-6.79813e+003, 1.24650e+003,-8.20622e+001,-2.33476e+000, 2.46459e-001],
            [3.64336e+005,-2.08457e+005, 4.33028e+004,-3.67825e+003, 4.22914e+001, 1.42701e+001],
            [-1.79298e+006, 1.01843e+006,-2.10037e+005, 1.82222e+004,-4.33573e+002,-2.72725e+001]
        ),
    ]

    if x_in_x0 > 0.2:
        logging.warn("try to approximate bethe heitler loss for x/x0={}".format(x_in_x0))

    x_in_x0 = min(x_in_x0, 0.2)

    def poly(coeffs):
        value = 0
        for c in coeffs:
            value = x_in_x0 * value + c;
        return value

    return [
        (
            1. / (1 + np.exp(-poly(weight_coeffs))),
            1. / (1 + np.exp(-poly(mean_coeffs))),
            np.exp(poly(var_coeffs))
        )
        for weight_coeffs, mean_coeffs, var_coeffs in kBetheHeitlerPolynomialApprox
    ]



def gaussian_mixture_moments(components):
    sum_w = 0
    dim = len(components[0][1])
    merged_mean = np.zeros(dim)

    for w, mean, cov in components:
        sum_w += w
        merged_mean += w*mean

    merged_mean /= sum_w

    merged_cov = np.zeros((dim,dim))
    for w, mean, cov in components:
        merged_cov += w * cov
        diff = (mean - merged_mean).reshape(1,-1)
        merged_cov += w * diff.transpose() @ diff

    merged_cov /= sum_w

    return merged_mean, merged_cov

def gaussian_mixture_mode(mixture):
    dists = [ (w, multivariate_normal(m, c)) for w, m, c in mixture ]

    def hessian(x):
        def H_m(w, d):
            a = w*d.pdf(x)
            r = (x - d.mean).reshape(-1,1)
            b = r@r.T - d.cov
            return a * (np.linalg.inv(d.cov) @ b @ np.linalg.inv(d.cov))

        return sum([ H_m(w,d) for w, d in dists ])

    def f(x):
        p_x = sum([ w*d.pdf(x) for w, d in dists ])

        a = sum([ w*d.pdf(x) / p_x * np.linalg.inv(d.cov) for w, d in dists ])
        b = sum([ w*d.pdf(x) / p_x * (np.linalg.inv(d.cov) @ d.mean) for w, d in dists ])

        return np.linalg.inv(a) @ b

    M = []

    tol = 1.e-8
    min_diff = 100*tol
    eigv_max = 0.01

    for m in mixture:
        x = m[1]

        for i in range(50):
            x_old = x
            x = f(x)

            if np.linalg.norm(x - x_old) < 1.e-8:
                break

        H = hessian(x)

        if max(np.linalg.eigvals(H)) < eigv_max:
            N = [ n for n in M if np.linalg.norm(n - x) < min_diff ] + [x]
            p_N = [ sum([w*d.pdf(n) for w, d in dists ]) for n in N ]
            M = [ n for n in M if np.linalg.norm(n - x) >= min_diff ] + [ N[np.argmax(p_N)] ]

    res = [ (m, sum([w*d.pdf(m) for w, d in dists ])) for m in M ]

    # sort by value, and reverse (since sort is smalles first)
    return sorted(res, reverse=True, key=lambda m: m[1])
