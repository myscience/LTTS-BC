import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

from time import sleep

class Reach:
    '''
        This represent the Reach environment. In this environment the task is to
        move a simple point to reach a predefined target in a 2D world. Actions
        are the possible moves in 2D worlds: up-down-left-right. At each time step
        the environment return the immediate reward and the observation space,
        which consists in the relative distances between current agent position
        and target. Environment also provides a flag: 'done', which signals
        wheter an episode is concluded, this can be for example occur when agent
        exits the bounds or target is reached or total simulation time expires.
    '''
    def __init__(self, max_T = 100, targ = None, init = None, extent = ((-1, -1), (1, 1)), render = True):
        # Here we collect environment duration and span
        self.max_T = max_T;
        self.extent = np.array (extent);
        self.scale = np.mean (self.extent[1] - self.extent[0])
        self.inv_scale = 1 / self.scale;

        # Keep internal time
        self.t = 0;

        # Here we init observation array and done flag
        self.obv = np.empty (2);
        self.done = False;

        # Here we init the position of target and agent
        self.targ = np.array (targ) if targ is not None else np.random.uniform (*extent);
        self.agen = np.array (init) if init is not None else np.ones (2) * 0.5;

        self.traj = np.empty ((2, max_T));
        self.traj [:, 0] = self.agen;

        # Here we init the reward to zero
        self.r = 0.;

        # Here we prepare for rendering
        self.do_render = render;


        if self.do_render:
            self.fig, self.ax = plt.subplots ();
            self.fig.tight_layout ();

            self.ax.set_xlim (self.extent[0][0] - 0.1, self.extent[1][0] + 0.1);
            self.ax.set_ylim (self.extent[0][1] - 0.1, self.extent[1][1] + 0.1);
            self.ax.axis ('off');

            # Bbox object around which the fancy box will be drawn.
            bb = mtransforms.Bbox(self.extent)
            domain = FancyBboxPatch((bb.xmin, bb.ymin),
                                    abs(bb.width), abs(bb.height),
                                    boxstyle="sawtooth,pad=.02",
                                    fc = 'none',
                                    ec = 'k',
                                    zorder = -10);

            self.ax.add_patch(domain);

            self.ptarg = self.ax.scatter (*self.targ, s = 500, marker = '*', color = 'gold');
            self.pagen = self.ax.scatter (*self.agen, s = 60, marker = 'p', color = 'darkred');

            self.ptraj, = self.ax.plot (*self.agen, c = 'r');
            self.time = self.ax.text (0.42, -0.1, 'Time 0', fontsize = 16);
            plt.ion();

    def step(self, action):
        # Here we apply the provided action to the agent state.
        self.agen += np.array (action);
        self.traj [:, self.t] = self.agen;

        dist = np.sqrt (np.square (self.targ - self.agen).mean ());

        self.obv [:] = self.targ - self.agen;
        self.r = self.dense_r (dist);

        self.done = dist < 0.01 * self.scale or self.t > self.max_T or self.out ();

        if self.do_render: self.render ();

        # Here we increase the env time
        self.t += 1

        return self.obv, self.r, self.done, self.agen;

    def render(self):
        self.ptraj.set_data (*self.traj[:, :self.t]);
        self.pagen.set_offsets (self.traj[:, self.t - 1]);

        self.time.set_text ('Time {}'.format (self.t));

        # Here we signal redraw
        self.fig.canvas.draw ();
        self.fig.canvas.flush_events();
        plt.show ();
        sleep (0.02);

        return self.fig;

    def out(self):
        return self.agen[0] < self.extent[0][0] or\
               self.agen[0] > self.extent[1][0] or\
               self.agen[1] < self.extent[0][1] or\
               self.agen[1] > self.extent[1][1];

    def dense_r(self, dist):
        # Dense reward is defined as decaying with the distance between agent
        # position and target.
        return np.exp (-dist * self.inv_scale);


    def build_expert (self, targ, init, steps = 80, T = 100, offT = 1, norm = True):
        assert T > steps;

        dx, dy = (targ - init) / steps;

        inp = targ - (init + np.array ([((i + 1) * dx, (i + 1) * dy) for i in range (steps)]))
        out = np.array ([[dx, dy] * steps]).reshape (-1, 2)

        inp = np.pad (inp, ((0, T - inp.shape [0]), (0, 0))).T;
        out = np.pad (out, ((0, T - out.shape [0]), (0, 0))).T;

        out[:, :offT] = 0;
        out[:, steps:] = 0

        if norm: out /= np.max (np.abs (out))

        return inp, out;
