import visualize as vs
import numpy as np
from ltts import LTTS
from env import Reach, Intercept

# Here we define our model
N, I, O, T = 150, 50, 2, 100;
shape = (N, I, O, T);

dt = 1 / T;
tau_m = 2. * dt;
tau_s = 2. * dt;
tau_ro = 5. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 4.;
sigma_input = 6.;
offT = 2;
dv = 1 / 5.;
alpha = .01;
alpha_rout = .1;
Vo = -4;
h = -4;
s_inh = 20;

# Here we build the dictionary of the simulation parameters
par = {'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'beta_ro' : beta_ro,
	   'dv' : dv, 'alpha' : alpha, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
	   'N' : N, 'T' : T, 'dt' : dt, 'offT' : offT, 'alpha_rout' : alpha_rout,
	   'sigma_input' : sigma_input, 'sigma_teach' : sigma_teach, 'shape' : shape};


# Here we init our model
ltts = LTTS (par);

# Here we init the environment
init = np.array ((0., 0.));
targ = np.array ((0., 1.));
vtargs = np.array ([(.2, 0.), (0.6, 0.)]);

env = Intercept (init = init, targ = targ, dt = dt);

# Here we ask the env for the expert behaviour
experts = [env.build_expert (targ, init, v, offT = 15) for v in vtargs];

# Here we implement these expert behaviour into spike patterns
itargs, inps = ltts.implement (experts);

# Here we clone these behaviour
ltts.clone (experts, itargs, epochs = 100);

# Here we save our model
# ltts.save ('model.npy');
# ltts.load ('model.npy')

agen = ltts.compute (inps[0]);
vs.cloning_plot ((itargs[0], experts[0][1]), agen, save = 'test0.jpeg');

agen = ltts.compute (inps[1]);
vs.cloning_plot ((itargs[1], experts[1][1]), agen, save = 'test1.jpeg');

# agen = ltts.compute (inps[2]);
# vs.cloning_plot ((itargs[2], experts[2][1]), agen, save = 'test2.jpeg');

# Here we test our model on different range of possible velocities
size = 100
vtests = np.array ([(vx, 0) for vx in np.linspace (0.1, 0.8, num = size)]);

Rs = np.ones (size) * 1e10;
hist = {'agent' : np.empty ((size, T, 2)), 'targ' : np.empty ((size, T, 2))};

for i, vtarg in enumerate (vtests):
	ltts.reset ();
	env.reset (init = init, targ = targ, vtarg = vtarg);
	obv = env.encode (targ - init)

	for t in range (T):
		action, S = ltts.step (obv, t);
		obv, r, done, agen = env.step (action);

		# print (agen)

		Rs[i] = min (r, Rs[i]);

		hist['agent'][i, t] = agen;
		hist['targ'][i, t] = env.targ;

		# env.render ();
		if done: break;

vs.reward_plot ((vtargs, vtests), Rs, save = 'test.jpeg');

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import kde

x, y = data = hist['agent'].T.reshape (2, -1)
x = x[~np.isnan (x)];
y = y[~np.isnan (y)];

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
KDE = kde.gaussian_kde([x, y])

nbins=300
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
zi = KDE(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
fig, ax = plt.subplots ();
img = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap = 'cividis', norm = LogNorm(vmin = 0.1));

# bins = 50;
#
# _, _, _, img = ax.hist2d (*data, bins = bins, norm = LogNorm(vmin = 1));

fig.colorbar (img, ax = ax)

# X, Y = np.mgrid[0:1:bins*1j, 0:1:bins*1j];
# Z, _, _ = np.histogram2d (hist['agent'][:, 0, :].ravel(),
# 						  hist['agent'][:, 1, :].ravel(),
# 						  bins = bins, range = ((0, 1), (0, 1)));
#
# ax.contourf (X, Y, Z, levels = 10);

fig.savefig ('hist.png');
