import visualize as vs
import numpy as np
from ltts import LTTS
from env import Reach

import matplotlib.pyplot as plt

# Here we define our model
N, I, O, T = 100, 2, 2, 200;

dt = 1 / T;
tau_m = 2. * dt;
tau_s = 2. * dt;
tau_ro = 5. * dt;
beta_s  = np.exp (-dt / tau_s);
beta_ro = np.exp (-dt / tau_ro);
sigma_teach = 5.;
sigma_input = 5.;
offT = 1;
dv = 1 / 5.;
alpha = .1;
alpha_rout = .1;
Vo = -4;
h = -4;
s_inh = 20;

# Here we build the dictionary of the simulation parameters
par = {'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'beta_ro' : beta_ro,
	   'dv' : dv, 'alpha' : alpha, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
	   'N' : N, 'T' : T, 'dt' : dt, 'offT' : offT, 'alpha_rout' : alpha_rout,
	   'sigma_input' : sigma_input, 'sigma_teach' : sigma_teach};

# Here we define target and initial position

for k in range(10):

	targ = np.array ((0.4, 0.3));
	init = np.array ((0.0, 0.0));

	# Here we init our model
	ltts = LTTS ((N, I, O, T), par);

	# Based on this information we compute the expert trajectory input-output and
	# produce a network behaviour to clone

	steps = 80;
	dx, dy = (targ - init) / steps;

	inp = targ - (init + np.array ([((i + 1) * dx, (i + 1) * dy) for i in range (steps)]))
	out = np.array ([[dx, dy] * steps]).reshape (-1, 2)

	inp = np.pad (inp, ((0, T - inp.shape [0]), (0, 0))).T;
	out = np.pad (out, ((0, T - out.shape [0]), (0, 0))).T;

	out[:, :offT] = 0;
	out[:, steps:] = 0

	inp /= np.max ( np.abs(inp))
	out /= np.max ( np.abs(out))

	out += np.random.uniform (-0.05, 0.05, size = out.shape);
	Inp = ltts.Jin @ inp + ltts.Jteach @ out;
	Targ, _ = ltts.compute (Inp);

	# Here we clone this behaviour
	Inp = ltts.Jin @ inp;
	ltts.clone ((Targ, out), Inp);

	####

	targ = np.array ((-0.4, -0.3));
	init = np.array ((0.0, 0.0));

	# Here we init our model
	#ltts = LTTS ((N, I, O, T), par);

	# Based on this information we compute the expert trajectory input-output and
	# produce a network behaviour to clone

	steps = 80;
	dx, dy = (targ - init) / steps;

	inp = targ - (init + np.array ([((i + 1) * dx, (i + 1) * dy) for i in range (steps)]))
	out = np.array ([[dx, dy] * steps]).reshape (-1, 2)

	inp = np.pad (inp, ((0, T - inp.shape [0]), (0, 0))).T;
	out = np.pad (out, ((0, T - out.shape [0]), (0, 0))).T;

	out[:, :offT] = 0;
	out[:, steps:] = 0

	inp /= np.max ( np.abs(inp))
	out /= np.max ( np.abs(out))

	out += np.random.uniform (-0.05, 0.05, size = out.shape);
	Inp = ltts.Jin @ inp + ltts.Jteach @ out;
	Targ, _ = ltts.compute (Inp);

	# Here we clone this behaviour
	Inp = ltts.Jin @ inp;
	ltts.clone ((Targ, out), Inp);


####

# Here we test the resulting behaviour

#targ = np.array ((.4, 0.3));

S_gen, action, pos_out = ltts.compute_online (init, targ);

vs.cloning_plot ((Targ, out), (S_gen, action), save = 'test-raster.eps');

#pos_out = np.zeros((2, T))
#pos_targ = np.zeros((2, T))

#pos_out[:,0] = init
#pos_targ[:,0] = init

#for t in range (T - 1):
#	pos_out[:,t+1] = pos_out[:,t] + out[:,t+1]*dt*.5
	#pos_targ[:,t+1] = pos_targ[:,t] + Targ*dt

S_gen, action, pos_out = ltts.compute_online (init, targ);
plt.plot(pos_out[0,:],pos_out[1,:])
plt.plot(targ[0],targ[1],'r.')

#plt.show()

# Here we move the target
# targ /= 1.1

# Here we init the environment

targ = np.array ((-0.4, -0.3+.1));
env = Reach (max_T = T, targ = targ, init = init);

traj = np.zeros((2, T-1))
dpos = np.zeros((2, T-1))

obv = init
for t in range (T - 1):
	action = ltts.step (obv * steps);
	obv, r, done, agent = env.step (action / steps);

	#fig = env.render ();
	traj[:,t] = agent
	dpos[:,t] = obv
	# fig.savefig ('test.jpeg');

plt.figure()
plt.plot(traj[0,:],traj[1,:])
plt.plot(targ[0],targ[1],'r.')

plt.figure()
plt.plot(traj[0,:])
plt.plot(traj[1,:])

plt.figure()
plt.plot(dpos[0,:])
plt.plot(dpos[1,:])


targ = np.array ((0.4, -0.3));
env = Reach (max_T = T, targ = targ, init = init);

ltts.t=0;
ltts.S = ltts.S*0;
ltts.S_hat = ltts.S_hat*0;

#T = 200;

#ltts.T = T;
#env.T = T;

traj = np.zeros((2, T-1))
dpos = np.zeros((2, T-1))

#ltts.S = np.zeros((N, T-1))
#ltts.S_hat = np.zeros((N, T-1))

obv = init
for t in range (T - 1):
	action = ltts.step (obv * steps);
	obv, r, done, agent = env.step (action / steps);

	#fig = env.render ();
	traj[:,t] = agent
	dpos[:,t] = obv
	# fig.savefig ('test.jpeg');

plt.figure()
plt.plot(traj[0,:],traj[1,:])
plt.plot(targ[0],targ[1],'r.')

plt.figure()
plt.plot(traj[0,:])
plt.plot(traj[1,:])

fig = plt.figure()
plt.plot(dpos[0,:])
plt.plot(dpos[1,:])

plt.show(fig)
