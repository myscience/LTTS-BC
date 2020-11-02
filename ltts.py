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

    def __init__ (self, shape, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = shape;
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
        self.h = h if isinstance (h, np.ndarray) else np.ones (net_shape) * h;

        # Membrane potential
        self.H = np.ones (net_shape) * par['Vo'];

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (net_shape);
        self.S_hat = np.zeros (net_shape);


        # This is the single-time output buffer
        self.out = np.zeros (self.N);

        # Here we save the params dictionary
        self.par = par;

        # Internal time keeping
        # self.t = 0;

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

    # def step (self, inp):
    #     t = self.t;
    #
    #     itau_m = self.itau_m;
    #     itau_s = self.itau_s;
    #
    #     self.S_hat [:, t] = (self.S_hat [:, t - 1] * itau_s + self.S [:, t] * (1. - itau_s)) if t > 0 else self.S_hat [:, 0];
    #
    #     self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + self.Jin @ inp + self.h [:, t])\
    #                                                       + self.Jreset @ self.S [:, t];
    #
    #     self.S [:, t + 1] = self._sigm (self.H [:, t + 1], dv = self.dv) - 0.5 > 0.;
    #
    #     # Here we use our policy to suggest a novel action given the system
    #     # current state
    #     action = self.policy ();
    #
    #     self.t += 1;
    #
    #     # Here we return the chosen next action
    #     return action

    # def policy (self):
    #     filt = self.S [:, self.t] * (1 - self.itau_ro)  + self.S[:, self.t + 1] * self.itau_ro;
    #
    #     return self.Jout @ filt;

    def step (self, inp, t):
        itau_m = self.itau_m;
        itau_s = self.itau_s;

        self.S_hat [:, t] = (self.S_hat [:, t - 1] * itau_s + self.S [:, t] * (1. - itau_s)) if t > 0 else self.S_hat [:, 0];

        self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + self.Jin @ inp + self.h [:, t])\
                                                          + self.Jreset @ self.S [:, t];

        self.S [:, t + 1] = self._sigm (self.H [:, t + 1], dv = self.dv) - 0.5 > 0.;

        # Here we use our policy to suggest a novel action given the system
        # current state
        action = self.policy (self.S[:, t + 1]);

        # Here we return the chosen next action
        return action

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

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        self.S [:, 0] = init if init else np.zeros (self.N);
        self.S_hat [:, 0] = self.S [:, 0] * itau_s;

        for t in range (self.T - 1):
            self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + self.S [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];

            self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (
                                                                ((self.J @ self.S_hat [:, t]) if rec else 0)
                                                                + inp [:, t] + self.h [:, t])\
                                                              + self.Jreset @ self.S [:, t];

            self.S [:, t + 1] = self._sigm (self.H [:, t + 1], dv = self.dv) - 0.5 > 0.;

        beta_ro = self.par['beta_ro'];
        return self.S.copy (), self.Jout @ ut.sfilter (self.S, itau = beta_ro);

    def implement (self, expert):
        if (self.J != 0).any ():
            print ('WARNING: Implement expert with non-zero recurrent weights\n');

        # First we parse expert input-output pair
        inp, out = expert

        # We then build a target network dynamics
        Inp = self.Jin @ inp;
        tInp = Inp + self.Jteach @ out;

        targ, _ = self.compute (tInp, rec = False);

        return targ, Inp;

    def compute_online (self, pos0, targ, init_s = None):
        # Check correct input shape
        #assert inp.shape[0] == self.N;

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        #self.S [:, 0] = init if init else np.zeros (self.N);
        self.S_hat [:, 0] = self.S [:, 0] * itau_s;

        Pos = np.zeros((2, self.T ))

        Pos [:, 0] = pos0;

        for t in range (self.T - 1):
            self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + self.S [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];

            vel = self.Jout @ self.S_hat [:, t];
            Pos [:, t+1] = Pos [:, t] + vel*self.dt;
            self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + self.Jin @ ( targ - Pos [:, t+1]) + self.h [:, t])\
                                                              + self.Jreset @ self.S [:, t];

            self.S [:, t + 1] = self._sigm (self.H [:, t + 1], dv = self.dv) - 0.5 > 0.;

        beta_ro = self.par['beta_ro'];
        return self.S.copy (), self.Jout @ ut.sfilter (self.S, itau = beta_ro), Pos;

    # def clone (self, Targ, Inp, epochs = 500):
    #     targ, out = Targ;
    #
    #     itau_m = self.itau_m;
    #     itau_s = self.itau_s;
    #
    #     alpha = self.par['alpha'];
    #     alpha_rout = self.par['alpha_rout'];
    #     beta_ro = self.par['beta_ro'];
    #
    #     self.S [:, 0] = targ [:, 0].copy ();
    #     self.S_hat [:, 0] = self.S [:, 0] * itau_s;
    #
    #     Tmax = np.shape (targ) [-1];
    #     dH = np.zeros ((self.N, self.T));
    #
    #     # Here we pre-train the readout matrix
    #     S_rout = ut.sfilter (targ, itau = beta_ro);
    #     adam = Adam (alpha = alpha_rout, drop = 0.9, drop_time = 50);
    #
    #     self.Jout = adam.optimize (ut.dJ_rout, out, S_rout, init = self.Jout, t_max = 1000);
    #
    #     # Here we train the network - online mode
    #     adam = Adam (alpha = alpha, drop = 0.9, drop_time = 100 * self.T);
    #
    #     for epoch in trange (epochs, leave = False, desc = 'Cloning'):
    #         # Here we train the network
    #         for t in range (self.T - 1):
    #             self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + targ [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];
    #
    #             self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + Inp [:, t] + self.h [:, t])\
    #                                                               + self.Jreset @ targ [:, t];
    #
    #             dH [:, t + 1] = dH [:, t]  * (1. - itau_m) + itau_m * self.S_hat [:, t];
    #
    #             dJ = np.outer (targ [:, t + 1] - self._sigm (self.H [:, t + 1], dv = self.dv), dH [:, t + 1]);
    #             self.J = adam.step (self.J, dJ);
    #
    #             np.fill_diagonal (self.J, 0.);
    #
    #         # if mode == 'offline':
    #         #     dJ = (targ - sigm (self.H, dv = dv)) @ dH.T;
    #         #     self.J = opt_rec.step (self.J, dJ);
    #         #
    #         #     np.fill_diagonal (self.J, 0.);
    #
    #     return ;

    def clone (self, experts, targets, epochs = 500):
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

        dH = np.zeros ((self.N, self.T));

        # Here we pre-train the readout matrix
        for expert, targ in zip (experts, targets):
            inp, out = expert;

            S_rout = ut.sfilter (targ, itau = beta_ro);

            adam = Adam (alpha = alpha_rout, drop = 0.9, drop_time = 50);
            self.Jout = adam.optimize (ut.dJ_rout, out, S_rout, init = self.Jout, t_max = 1000);

        # Here we train the network - online mode
        adam = Adam (alpha = alpha, drop = 0.9, drop_time = 100 * self.T);

        targets = np.array (targets)
        inps = np.array ([self.Jin @ exp[0] for exp in experts]);
        outs = np.array ([exp[1] for exp in experts]);

        self.S [:, 0] = targ [:, 0].copy ();
        self.S_hat [:, 0] = self.S [:, 0] * itau_s;

        # Here we train the network
        for epoch in trange (epochs, leave = False, desc = 'Cloning'):
            shuffle ((inps, outs, targets));

            for inp, out, targ in zip (inps, outs, targets):

                for t in range (self.T - 1):
                    self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + targ [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];

                    self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + inp [:, t] + self.h [:, t])\
                                                                      + self.Jreset @ targ [:, t];

                    dH [:, t + 1] = dH [:, t]  * (1. - itau_m) + itau_m * self.S_hat [:, t];

                    dJ = np.outer (targ [:, t + 1] - self._sigm (self.H [:, t + 1], dv = self.dv), dH [:, t + 1]);
                    self.J = adam.step (self.J, dJ);

                    np.fill_diagonal (self.J, 0.);

        return ;


    def train (self, targ, inp, mode = 'online', epochs = 500, par = None,
                out = None, Jrout = None, track = False):
        '''
            This is the main function of the model: is used to trained the system
            given a target and and input. Two training mode can be selected:
            (offline, online). The first uses the exact likelihood gradient (which
            is non local in time, thus non biologically-plausible), the second is
            the online approx. of the gradient as descrived in the article.
            Args:
                targ: numpy.array of shape (N, T), where N is the number of neurons
                      in the network and T is the time length of the sequence.
                inp : numpy.array of shape (N, T) that collects the input signal
                      to neurons.
            Keywords:
                mode : (default: online) The training mode to use, either 'offline'
                       or 'online'.
                epochs: (default: 500) The number of epochs of training.
                par   : (default: None) Optional different dictionary collecting
                        training parameters: {dv, alpha, alpha_rout, beta_ro, offT}.
                        If not provided defaults to the parameter dictionary of
                        the model.
                out   : (default: None) Output target trajectories, numpy.array
                        of shape (K, T), where K is the dimension of the output
                        trajectories. This parameter should be specified if either
                        Jrout != None or track is True.
                Jrout : (default: None) Pre-trained readout connection matrix.
                        If not provided, a novel matrix is built and trained
                        simultaneously with the recurrent connections training.
                        If Jrout is provided, the out parameter should be specified
                        as it is needed to compute output error.
                track : (default: None) Flag to signal whether to track the evolution
                        of output MSE over training epochs. If track is True then
                        the out parameters should be specified as it is needed to
                        compute output error.
        '''
        assert (targ.shape == inp.shape);

        par = self.par if par is None else par;

        dv = par['dv'];

        itau_m = self.itau_m;
        itau_s = self.itau_s;

        sigm = self._sigm;

        alpha = par['alpha'];
        alpha_rout = par['alpha_rout'];
        beta_ro = par['beta_ro'];
        offT = par['offT'];

        self.S [:, 0] = targ [:, 0].copy ();
        self.S_hat [:, 0] = self.S [:, 0] * itau_s;

        Tmax = np.shape (targ) [-1];
        dH = np.zeros ((self.N, self.T));

        track = np.zeros (epochs) if track else None;

        opt_rec = Adam (alpha = alpha, drop = 0.9, drop_time = 100 * Tmax if mode == 'online' else 100);

        if Jrout is None:
            S_rout = ut.sfilter (targ, itau = beta_ro);
            J_rout = np.random.normal (0., 0.1, size = (out.shape[0], self.N));
            opt = Adam (alpha = alpha_rout, drop = 0.9, drop_time = 20 * Tmax if mode == 'online' else 20);

        else:
            J_rout = Jrout;

        for epoch in trange (epochs, leave = False, desc = 'Training {}'.format (mode)):
            if Jrout is None:
                # Here we train the readout
                dJrout = (out - J_rout @ S_rout) @ S_rout.T;
                J_rout = opt.step (J_rout, dJrout);

            # Here we train the network
            for t in range (Tmax - 1):
                self.S_hat [:, t] = self.S_hat [:, t - 1] * itau_s + targ [:, t] * (1. - itau_s) if t > 0 else self.S_hat [:, 0];

                self.H [:, t + 1] = self.H [:, t] * (1. - itau_m) + itau_m * (self.J @ self.S_hat [:, t] + inp [:, t] + self.h [:, t])\
                                                                  + self.Jreset @ targ [:, t];

                dH [:, t + 1] = dH [:, t]  * (1. - itau_m) + itau_m * self.S_hat [:, t];

                if mode == 'online':
                    dJ = np.outer (targ [:, t + 1] - sigm (self.H [:, t + 1], dv = dv), dH [:, t + 1]);
                    self.J = opt_rec.step (self.J, dJ);

                    np.fill_diagonal (self.J, 0.);

            if mode == 'offline':
                dJ = (targ - sigm (self.H, dv = dv)) @ dH.T;
                self.J = opt_rec.step (self.J, dJ);

                np.fill_diagonal (self.J, 0.);

            # Here we track MSE
            if track is not None:
                S_gen = self.compute (inp, init = np.zeros (self.N));
                track [epoch] = np.mean ((out - J_rout @ ut.sfilter (S_gen,
                                                itau = beta_ro))[:, offT:]**2.);

        return (J_rout, track) if Jrout is None else track;
