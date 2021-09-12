from main import *
import pandas as pd

pgf = True

if pgf:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

lambdas = [0.1, 0.2, 0.5, 1, 2, 5, 10]
ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

df = pd.DataFrame(data=None, index=ps, columns=lambdas)

for lamb in lambdas:
    vals = []
    for p in ps:
        vals.append(core.calculate_z_length(p=p, lamb=lamb))

    df.iloc[:, lambdas.index(lamb)] = vals
    print(df)


plt.imshow(df, cmap='plasma')
plt.xlabel('$\lambda$')
plt.ylabel('$p$')
plt.colorbar()
plt.xticks(range(len(lambdas)), df.columns)
plt.yticks(range(len(ps)), df.index)
plt.savefig(f'report/figures/hmap.pgf') if pgf else plt.show()
plt.close(fig='all')