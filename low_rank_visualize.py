import visualize as vs
import numpy as np
import utils as ut
from ltts import LTTS
from env import Unlock

from itertools import product as Prod

# Here we define our model
N, I, O, T = 100, 5, 3, 50;
shape = (N, I, O, T);

dt = 1 / T;
tau_m = 8. * dt;
tau_s = 2. * dt;
tau_ro = 10. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 5.;
sigma_input = 8.;
sigma_hint  = 5.;
offT = 5;
dv = 1 / 20.;
alpha = 5.;
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

P = ut.kTrajectory (T, offT = offT, norm = True)#[:1, :T];
C = ut.kClock (T);

experts = [(C, P)];
targ, inp = ltts.implement (experts);

# Traj = [];
# from tqdm import trange
# for i in trange (1, leave = False):
	# ltts.J = np.random.normal (0., 3. / np.sqrt(N), size = (N, N));
	# Traj.append (ltts.clone (experts, targ, epochs = 1000, rank = None 	));

track = ltts.clone (experts, targ, epochs = 1000, rank = 90);

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

S, out = ltts.compute (inp[0]);
plt.plot (P.T, c = 'r', ls = '--')
plt.plot (out.T);
plt.show ();
plt.plot (track);
plt.show ();
plt.imshow ((S - targ[0]), aspect = 'auto', cmap = 'binary')
plt.show ();

assert False

def colored_line (canvas, data, cmap = 'winter', cb = None, proj = False):
    fig, ax = canvas;
    traj = np.array (data).reshape (-1, 1, 3)

    T = traj.shape [0];

    cmap = cm.get_cmap(cmap);
    norm = plt.Normalize (0, T);
    t = np.linspace (0, T, num = T);
    c = np.array ([cmap (i**(1/4)) for i in np.linspace (0., 1., num = T)]);
    c[:, -1] = np.linspace (1., 0.05, num = T)**2;

    traj = np.concatenate ((traj[:-1], traj[1:]), axis = 1);
    lc = Line3DCollection (traj, colors = c);
    line = ax.add_collection3d (lc);

    if proj:
        xvec = np.array ([0., 1., 1.]), np.array ([ax.get_xlim()[0], 0, 0]);
        yvec = np.array ([1., 0., 1.]), np.array ([0, ax.get_ylim()[0], 0]);
        zvec = np.array ([1., 1., 0.]), np.array ([0, 0, ax.get_zlim()[0]]);

        xtraj = np.concatenate ((traj[:-1] * xvec[0] + xvec[1],
                                 traj[1:]  * xvec[0] + xvec[1]), axis = 1)
        ytraj = np.concatenate ((traj[:-1] * yvec[0] + yvec[1],
                                 traj[1:]  * yvec[0] + yvec[1]), axis = 1)
        ztraj = np.concatenate ((traj[:-1] * zvec[0] + zvec[1],
                                 traj[1:]  * zvec[0] + zvec[1]), axis = 1)

        xlc = Line3DCollection (xtraj, colors = c, alpha = 0.3);
        ylc = Line3DCollection (ytraj, colors = c, alpha = 0.3);
        zlc = Line3DCollection (ztraj, colors = c, alpha = 0.3);

        line = ax.add_collection3d (xlc);
        line = ax.add_collection3d (ylc);
        line = ax.add_collection3d (zlc);

    ax.scatter (*data[-1], color = 'r', marker = 'o', s = 20);


    if cb:
        cax_a = fig.add_axes(cb)
        cbax_a = fig.colorbar (line, cax = cax_a, orientation = 'horizontal', ticks = [0, T])
        cbax_a.ax.set_xticklabels ([0, 'T']);

pca = PCA (n_components = 4);

pca.fit (Traj[0]);
Dec = np.array ([pca.transform (traj) for traj in Traj]);

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim ((100, -100))
ax.set_ylim ((100, -100))
ax.set_zlim ((100, -100))

xlim = np.min (Dec[:, :, 0]), np.max (Dec[:, :, 0]);
ylim = np.min (Dec[:, :, 1]), np.max (Dec[:, :, 1]);
zlim = np.min (Dec[:, :, 2]), np.max (Dec[:, :, 2]);

def sgn (x): return 1 if x > 0 else -1;

xlim = xlim[0] - sgn(xlim[0]) * xlim[0] / 3, xlim[1] + xlim[1] / 10;
ylim = ylim[0] - sgn(ylim[0]) * ylim[0] / 3, ylim[1] + ylim[1] / 10;
zlim = zlim[0] - sgn(zlim[0]) * zlim[0] / 3, zlim[1] + zlim[1] / 10;

ax.set_xlim (xlim);
ax.set_ylim (ylim);
ax.set_zlim (zlim);

[colored_line ((fig, ax), dec[:, :3], proj = False) for dec in Dec];

# Make the panes transparent
ax.xaxis.set_pane_color ((1.0, 1.0, 1.0, 0.0));
ax.yaxis.set_pane_color ((1.0, 1.0, 1.0, 0.0));
ax.zaxis.set_pane_color ((1.0, 1.0, 1.0, 0.0));

ax.set_xticklabels ([]);
ax.set_yticklabels ([]);
ax.set_zticklabels ([]);

ax.set_xlabel ('PCA 1')
ax.set_ylabel ('PCA 2')
ax.set_zlabel ('PCA 3')

fig.tight_layout ();

plt.show ();
assert False
save = 'frames/'
for t in trange (360, desc = 'Rendering', leave = False):
    ax.view_init (elev = 30., azim = 45 + t);
    fig.savefig (save + str(t).zfill(3) + '.png');
