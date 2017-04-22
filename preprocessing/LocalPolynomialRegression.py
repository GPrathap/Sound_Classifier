from py_qt import bootstrap as bs
import matplotlib.pyplot as plt
from py_qt import npr_methods
import numpy as np

from py_qt import nonparam_regression as smooth
from py_qt import plot_fit


def f(x):
    return 3*np.cos(x/2) + x**2/5 + 3

xs = np.random.rand(200) * 10
ys = f(xs) + 2*np.random.randn(*xs.shape)

grid = np.r_[0:10:512j]

plt.plot(grid, f(grid), 'r--', label='Reference')
plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
# plt.legend(loc='best')
# plt.show()

k0 = smooth.NonParamRegression(xs, ys, method=npr_methods.SpatialAverage())
k0.fit()
plt.plot(grid, k0(grid), label="Spatial Averaging", linewidth=2)
plt.legend(loc='best')

k1 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=1))
k2 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
k3 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=3))
k12 = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=12))
k1.fit(); k2.fit(); k3.fit(); k12.fit()
plt.figure()
plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
plt.plot(grid, k12(grid), 'b', label='polynom order 12', linewidth=2)
plt.plot(grid, k3(grid), 'y', label='cubic', linewidth=2)
plt.plot(grid, k2(grid), 'k', label='quadratic', linewidth=2)
plt.plot(grid, k1(grid), 'g', label='linear', linewidth=2)
plt.plot(grid, f(grid), 'r--', label='Target', linewidth=2)
plt.legend(loc='best')

yopts = k2(xs)
res = ys - yopts
plot_fit.plot_residual_tests(xs, yopts, res, 'Local Quadratic')

yopts = k1(xs)
res = ys - yopts
plot_fit.plot_residual_tests(xs, yopts, res, 'Local Linear')
plt.show()

# def fit(xs, ys):
#     est = smooth.NonParamRegression(xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
#     est.fit()
#     return est
#
# result = bs.bootstrap(fit, xs, ys, eval_points = grid, CI = (95,99))
#
# plt.plot(xs, ys, 'o', alpha=0.5, label='Data')
# plt.plot(grid, result.y_fit(grid), 'r', label="Fitted curve", linewidth=2)
# plt.plot(grid, result.CIs[0][0,0], 'g--', label='95% CI', linewidth=2)
# plt.plot(grid, result.CIs[0][0,1], 'g--', linewidth=2)
# plt.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='g', alpha=0.25)
# plt.legend(loc=0)
# plt.show()



