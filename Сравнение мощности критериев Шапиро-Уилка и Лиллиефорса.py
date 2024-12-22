from datetime import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import shapiro, norm, expon
from statsmodels.stats.diagnostic import lilliefors
import matplotlib.pyplot as plt

norm_sample = norm(loc=2, scale=3).rvs(1000)
exp_sample = expon(loc=2, scale=10).rvs(1000)
t_sample = np.random.standard_t(10, 1000)

def check_criterions(x, alpha):
    power_1 = []
    power_2 = []
    for N in x:
        rej_cnt_shapiro = 0
        rej_cnt_lillie = 0
        N = int(N)
        for _ in range(N):
            test = np.random.standard_t(10, N)

            pvalue_shapiro = shapiro(test).pvalue
            Dn, pvalue_lillie = lilliefors(test, dist='norm', pvalmethod='table')

            rej_cnt_shapiro += (pvalue_shapiro < alpha)
            rej_cnt_lillie += (pvalue_lillie < alpha)

        power_1.append(round(rej_cnt_shapiro / N, 8))
        power_2.append(round(rej_cnt_lillie / N, 8))
    return power_1, power_2

def show_plot(x, power_1, power_2):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.title(f'Сравнение мощности стат критериев Шапиро-Уилка и Лиллиефорса')
    plt.plot(x, power_1, 'g', label='Шапиро-Уилк')
    plt.plot(x, power_2, 'b', label='Лиллиефорс')
    plt.axhline(y=0.8, color='r')
    plt.legend()
    plt.grid(True, linestyle = '--', linewidth = 0.5)
    plt.show()

x = list(range(100, 5100, 200))#np.linspace(100, 10000, 10)
power_sh, power_lillie = check_criterions(x, alpha=0.01)
show_plot(x, power_sh, power_lillie)

# print(f"TPR или мощность для shapiro: {round(rej_cnt_shapiro / N, 8)}")
# print(f"TPR или мощность для lilliefors: {round(rej_cnt_lillie / N, 8)}")
