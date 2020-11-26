import visualize as vs
import numpy as np
from ltts import LTTS
from env import Unlock

from itertools import product as Prod

# Here we define our model
N, I, O, T = 150, 80, 2, 200;
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
btn = np.array ((0.2, .5));

# vtargs = np.random.uniform (-0.6, 0.6, size = (10, 2))
btns = np.array (((0.2, 0.5), ));

# btns = np.array ([(vx, vy) for vx, vy in Prod (np.linspace (-0.6, 0.6, num = 4),
# 											     np.linspace (-0.6, 0.6, num = 4))])

env = Unlock (init = init, targ = targ, btn = btn, dt = dt, res = 20);

# Here we ask the env for the expert behaviour
experts = [env.build_expert (targ, init, btn, offT = (15, 15)) for btn in btns];

# Here we implement these expert behaviour into spike patterns
itargs, inps = ltts.implement (experts);

# Here we clone these behaviour
ltts.clone (experts, itargs, epochs = 500);

# Here we save our model
ltts.save ('model.npy');
# ltts.load ('model.npy')

agen = ltts.compute (inps[0]);
vs.cloning_plot ((itargs[0], experts[0][1]), agen, save = 'test0.png');

# agen = ltts.compute (inps[1]);
# vs.cloning_plot ((itargs[1], experts[1][1]), agen, save = 'test1.jpeg');

# agen = ltts.compute (inps[2]);
# vs.cloning_plot ((itargs[2], experts[2][1]), agen, save = 'test2.jpeg');

#Rs = np.ones (size**2) * 1e10;
hist = {'agent' : np.zeros ((T, 2)), 'targ' : np.zeros ((T, 2)), 'T' : T};

ltts.reset ();

save = 'frames/'
obv = np.hstack ((env.encode (targ - init), env.encode (btn - init)));
for t in range (T):
	action, S = ltts.step (obv, t);
	obv, r, done, agen = env.step (action);

	# Rs[i] = min (r, Rs[i]);

	hist['agent'][t] = agen;
	hist['targ'][t] = env.targ;

	env.buffer ((S[:100], action));
	env.render (save = save);
	if done: break;

hist['agent'][t:] = agen;
hist['targ'][t:] = env.targ;

# vs.env_hist_plot (hist, save = 'test.jpeg');

# vs.reward_plot_2D ((vtargs, vtests), Rs, (size, size), save = 'test.jpeg');
# vs.trajectory_plot (hist, save = 'traj.png');
