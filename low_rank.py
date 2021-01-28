import visualize as vs
import numpy as np
import utils as ut
from ltts import LTTS
from env import Unlock

from itertools import product as Prod
from tqdm import tqdm, trange

# Here we define our model
N, I, O, T = 100, 80, 2, 65;
shape = (N, I, O, T);

dt = 1 / T;
tau_m = 6. * dt;
tau_s = 2. * dt;
tau_ro = 10. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 5.;
sigma_input = 8.;
sigma_hint  = 5.;
offT = 5;
dv = 1 / 5.;
alpha = 1.;
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
btn = np.array ((0., .0));

# --------- ROUND TASK ---------
rt, rb = .7, 0.;
Theta = np.array ((20, 60)) * np.pi / 180.;
tars = np.array ([(rt * np.cos (t), rt * np.sin (t)) for t in Theta]);
btns = [(.0, 0.)] * len(Theta)# np.array ([(rb * np.cos (t), rb * np.sin (t)) for t in Theta]);

env = Unlock (init = init, targ = targ, btn = btn, dt = dt, res = 20);

# Here we ask the env for the expert behaviour
experts = [env.build_expert (targ, init, btn, offT = (1, 5), steps = (4, 55), T = (5, 60))
			for targ, btn in zip (tars, btns)];

itargs, inps = ltts.implement (experts);

# ------------------------------
def round_test (ltts, env, size = 100, r = (1., -.3)):
	rt, rb = r;
	Rs = np.ones (size) * 1e10;
	Theta = np.linspace (20., 60., num = size) * np.pi / 180.;

	hist = {'agent' : np.zeros ((size, T, 2)), 'targ' : np.zeros ((size, T, 2)), 'T' : T};

	tars = np.array ([(rt * np.cos (t), rt * np.sin (t)) for t in Theta]);
	btns = [(0., 0.)] * len (Theta);#np.array ([(rb * np.cos (t), rb * np.sin (t)) for t in Theta]);

	for i, (targ, btn) in enumerate (zip (tars, btns)):
		env.reset (init = init, targ = targ, btn = btn);
		ltts.reset ();

		obv = np.hstack ((env.encode (targ - init), env.encode (btn - init)));
		for t in range (T):
			action, S = ltts.step (obv, t);
			obv, r, done, agen = env.step (action);

			Rs[i] = min (Rs[i], 5. if env.locked else r);

			hist['agent'][i, t] = agen;
			hist['targ'][i, t] = env.targ;

			# env.buffer ((S[:100], action));
			# env.render (save = None);
			if done: break;

		hist['agent'][i, t:] = agen;
		hist['targ'][i, t:] = env.targ;

	return Theta, hist, Rs;

# ranks = np.linspace (N // 2, N, num = 2, dtype = np.int);
ranks = np.linspace (N // 2, N, num = N // 2 + 1, dtype = np.int);
Data = [];

for rank in tqdm (ranks, desc = 'Scanning Ranks', leave = False):
	ltts.J = np.zeros ((N, N));
	ltts.Jout = np.zeros ((O, N));

	# Here we clone these behaviour
	ltts.clone (experts, itargs, epochs = 1000, rank = rank);

	# Here we save our model
	# ltts.save ('model.npy');
	# ltts.load ('model.npy')

	# agen = ltts.compute (inps[0]);
	# vs.cloning_plot ((itargs[0], experts[0][1]), agen, save = 'test0.jpeg');

	Data.append (round_test (ltts, env, r = (rt, rb)));

import pickle
with open ('Low_Rank.pkl', 'wb') as f:
	pickle.dump ([Theta, Data], f);
