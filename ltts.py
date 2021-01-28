'''
    This is the Learning Through Target Spikes (LTTS) repository for code associated to
    the paper: Paolo Muratore, Cristiano Capone, Pier Stanislao Paolucci (2020)
    "Target spike patterns enable efficient and biologically plausible learning for
     complex temporal tasks*" (currently *under review*).
    Please give credit to this paper if you use or modify the code in a derivative work.
    This work is licensed under the Creative Commons Attribution 4.0 International License.
    To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
    or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
'''

import numpy as np
import utils as ut
from tqdm import trange
from optimizer import Adam

class LTTS:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
    """

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape'];
        net_shape = (self.N, self.T);

        self.dt = 1. / self.T;
        self.itau_m = self.dt / par['tau_m'];
        self.itau_s = np.exp (-self.dt / par['tau_s']);
        self.itau_ro = np.exp (-self.dt / par['tau_ro']);

        self.dv = par['dv'];

        # This is the network connectivity matrix
        self.J = np.zeros ((self.N, self.N));

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I));
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O));

        self.Jout = np.zeros ((self.O, self.N));

        # Remove self-connections
        np.fill_diagonal (self.J, 0.);

        # Impose reset after spike
        self.s_inh = -par['s_inh'];
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh);

        # This is the external field
        h = par['h'];

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h;

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo'];
        self.Vo = par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N);
        self.S_hat = np.zeros (self.N);

        # This is the single-time output buffer
        self.out = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv;

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0;

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv;

        out = np.zeros (x.shape);
        mask = x > 0;
        out [mask] = 1. / (1. + np.exp (-y [mask]));
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]));

        return out;

    def step (self, inp, t):
        itau_m = self.itau_m;
        itau_s = self.itau_s;

        self.S_hat [:] = (self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s))

        self.H [:] = self.H [:] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:] + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S [:];

        self.S [:] = self._sigm (self.H [:], dv = self.dv) - 0.5 > 0.;

        # Here we use our policy to suggest a novel action given the system
        # current state
        action = self.policy (self.S);

        # Here we return the chosen next action
        return action, self.S.copy ()

    def policy (self, state):
        self.out = self.out * self.itau_ro  + state * (1 - self.itau_ro);

        return self.Jout @ self.out;

    def compute (self, inp, init = None, rec = True):
        '''
            This function is used to compute the output of our model given an
            input.
            Args:
                inp : numpy.array of shape (N, T), where N is the number of neurons
                      in the network and T is the time length of the input.
                init: numpy.array of shape (N, ), where N is the number of
                      neurons in the network. It defines the initial condition on
                      the spikes. Should be in range [0, 1]. Continous values are
                      casted to boolean.
            Keywords:
                Tmax: (default: None) Optional time legth of the produced output.
                      If not provided default is self.T
                Vo  : (default: None) Optional initial condition for the neurons
                      membrane potential. If not provided defaults to external
                      field h for all neurons.
                dv  : (default: None) Optional different value for the dv param
                      to compute the sigmoid activation.
        '''
        # Check correct input shape
        assert inp.shape[0] == self.N;

        N, T = inp.shape;

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        self.reset (init);

        Sout = np.zeros ((N, T));

        for t in range (T - 1):
            self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)

            self.H [:] = self.H * (1. - itau_m) + itau_m * ((self.J @ self.S_hat [:] if rec else 0)
                                                            + inp [:, t] + self.h)\
                                                         + self.Jreset @ self.S;

            self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0.;
            Sout [:, t] = self.S.copy ();

        return Sout, self.Jout @ ut.sfilter (Sout, itau = self.par['beta_ro']);

    def implement (self, experts, adapt = True):
        if (self.J != 0).any ():
            print ('WARNING: Implement expert with non-zero recurrent weights\n');

        # First we parse experts input-output pair
        inps = np.array ([exp[0] for exp in experts]);
        outs = np.array ([exp[1] for exp in experts]);

        # Here we extract the maxima of input and output which are used to balance
        # the signal injection using the sigma_input and sigma_teach variables
        if adapt:
            self.sigma_input = 5. / np.max (np.abs (inps));
            self.sigma_teach = 5. / np.max (np.abs (outs));

            self.par['sigma_input'] = self.sigma_input;
            self.par['sigma_teach'] = self.sigma_teach;

            self.Jin = np.random.normal (0., self.sigma_input, size = (self.N, self.I));
            self.Jteach = np.random.normal (0., self.sigma_teach, size = (self.N, self.O));

        # We then build a target network dynamics
        Inp = np.einsum ('ij, njt->nit', self.Jin, inps);
        tInp = Inp + np.einsum ('ij, njt->nit', self.Jteach, outs);

        Targ = [self.compute (t_inp, rec = False)[0] for t_inp in tInp];

        return Targ, Inp;

    def clone (self, experts, targets, epochs = 500, rank = None):
        assert len (experts) == len (targets);

        def shuffle(iter):
            rng_state = np.random.get_state();

            for a in iter:
                np.random.shuffle(a);
                np.random.set_state(rng_state);

        # Here we clone this behaviour
        itau_m = self.itau_m;
        itau_s = self.itau_s;

        alpha = self.par['alpha'];
        alpha_rout = self.par['alpha_rout'];
        beta_ro = self.par['beta_ro'];

        S_rout = [ut.sfilter (targ, itau = beta_ro) for targ in targets];
        adam_out = Adam (alpha = alpha_rout, drop = 0.9, drop_time = epochs // 10);

        # Here we train the network - online mode
        adam = Adam (alpha = alpha, drop = 0.9, drop_time = epochs // 10 * self.T);

        targets = np.array (targets)
        inps = np.array ([self.Jin @ exp[0] for exp in experts]);
        outs = np.array ([exp[1] for exp in experts]);

        dH = np.zeros (self.N);

        # FIXME: THIS IS THE RANK CODE PART: CHECK THIS!
        if rank:
            B = np.random.normal (0.,  1. / np.sqrt(rank), size = (rank, self.N));
            D = B.T @ B;
            D /= np.max (np.abs (D));

        # Here we train the network
        for epoch in trange (epochs, leave = False, desc = 'Cloning'):
            shuffle ((inps, outs, targets, S_rout));

            for out, s_out in zip (outs, S_rout):
                dJ = (out - self.Jout @ s_out) @ s_out.T;
                self.Jout = adam_out.step (self.Jout, dJ);

            for inp, out, targ in zip (inps, outs, targets):
                self.reset ();

                dH *= 0;

                for t in range (self.T - 1):
                    self.S_hat [:] = self.S_hat * itau_s + targ [:, t] * (1. - itau_s)

                    self.H [:] = self.H * (1. - itau_m) + itau_m * (self.J @ self.S_hat + inp [:, t] + self.h)\
                                                                 + self.Jreset @ targ [:, t];

                    dH [:] = dH  * (1. - itau_m) + itau_m * self.S_hat;

                    # dJ = np.outer (targ [:, t + 1] - self._sigm (self.H, dv = self.dv), dH);

                    # CHECK THIS: Compute DJ based on rank!
                    if rank:
                        # NOTE: _dsigm should be the psudo-derivative. Maybe this
                        #       should be set to zero!
                        dJ = np.outer (D @ (targ [:, t] - self._sigm(self.H, dv = self.dv)) +
                                       self._dsigm (self.H, dv = 1), dH);
                    else:
                        dJ = np.outer (targ [:, t] - self._sigm (self.H, dv = self.dv), dH);

                    self.J = adam.step (self.J, dJ);
                    np.fill_diagonal (self.J, 0.);

                # NOTE: MAYBE ONE SHOULD ADD HERE THE EVALUATION OFF THE OUTPUT
                #       ERROR. LIKE THIS
                # for out, s_out in zip(outs, S_rout):
                #     # s_out = ut.sfilter (Sout, itau = self.par['beta_ro']);
                #
                #     dJ = (out - self.Jout @ s_out) @ s_out.T;
                #     self.Jout = adam_out.step (self.Jout, dJ);
                #
                # s, p = self.compute (inp);
                # track[epoch] = np.sum (np.abs(targ - s));

        return ;

    def reset (self, init = None):
        self.S [:] = init if init else np.zeros (self.N);
        self.S_hat [:] = self.S [:] * self.itau_s if init else np.zeros (self.N);

        self.out [:] *= 0;

        self.H [:] *= 0.;
        self.H [:] += self.Vo;


    def save (self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jin, self.Jteach, self.Jout, self.J, self.par)

        np.save (filename, np.array (data_bundle, dtype = np.object));

    @classmethod
    def load (cls, filename):
        data_bundle = np.load (filename, allow_pickle = True);

        Jin, Jteach, Jout, J, par = data_bundle;

        obj = LTTS (par);

        obj.Jin = Jin.copy ();
        obj.Jteach = Jteach.copy ();
        obj.Jout = Jout.copy ();
        obj.J = J.copy ();

        return obj;
