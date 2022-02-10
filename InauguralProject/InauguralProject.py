from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

# THIS IS A MERGE TEST LINE
# NEW TEST LINE

mp = {"phi": 0.3, "epsilon": 0.5, "r": 0.03, "tau_g": 0.012, "tau_p": 0.004, "p_bar": 3}

def util(c, h, mp):
    return (c ** (1 - mp["phi"]) * h ** mp["phi"])

def opt_fun(h, m, mp):
    c = cons(m, h, mp)
    return -util(c, h, mp)

def tau(h, mp):
    p_tilde = h * mp["epsilon"]
    return mp["r"] * h + mp["tau_g"] * p_tilde + mp["tau_p"] * max(p_tilde - mp["p_bar"], 0)

def cons(m, h, mp):
    return m - tau(h, mp)

def solve(m, mp):
    sol = optimize.minimize_scalar(opt_fun, args = (m, mp))
    return sol.x

h = solve(0.5, mp)
print(h)
print(cons(0.5, h, mp))

opt_h = []
opt_c = []

ms = np.linspace(0.4, 1.5, 100)

for m in ms:
    h = solve(m, mp)
    c = cons(m, h, mp)
    opt_h.append(h)
    opt_c.append(c)

"""
fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax.plot(ms, opt_h)
ax.set_title('optimal housing, $h^{\star}$')
ax.set_xlabel('$m$')

ax = fig.add_subplot(1,2,2)
ax.plot(ms, opt_c)
ax.set_title('optimal consumption, $c^{\star}$')
ax.set_xlabel('$m$')

#plt.show()
"""
#3

np.random.seed(1)

m_i = np.random.lognormal(-0.4, 0.35, 10000)
print(m_i)

def tax_revenue(ms, mp):
    sum = []
    hs = []
    for m in ms:
        h = solve(m, mp)
        p_tilde = h * mp["epsilon"]
        sum.append((mp["tau_g"] * p_tilde + mp["tau_p"] * max(p_tilde - mp["p_bar"], 0)))
        hs.append(h)
    return np.array(sum), hs
tax_revenues, hs = tax_revenue(m_i, mp)

mean = tax_revenues.mean()
#print(f'Average housing tax revenue (1000s): {tax_revenues.mean()*1e3:2.3f}')

#plt.hist(tax_revenues, bins= 100)
#plt.show()

def objective(tau_g, tax_target, ms, mp):
    mp["tau_g"] = tau_g
    tax_revenues, hs = tax_revenue(ms, mp)
    return tax_target - tax_revenues.mean()

mp_new = {'phi': 0.3, 'epsilon': 0.8, 'r': 0.03, 'tau_g': 0.01, 'tau_p': 0.009, 'p_bar': 8}

target = mean
tau_0 = 0.005
tax_reform_sol = optimize.root(objective, tau_0, args=(target, m_i, mp_new))

print(tax_reform_sol.x)

mp_new['tau_g'] = tax_reform_sol.x
tax_post, hs = tax_revenue(m_i, mp_new)
print(tax_post.mean() - mean)

