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
    def __init__(self, targ = None, init = None, max_T = 500,  extent = ((-1, -1), (1, 1)), render = True):
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
        self.agen = np.array (init) if init is not None else np.zeros (2);

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

            self.time = self.ax.text (-0.2, 1.1, 'Time 0', fontsize = 16);
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

    def set_target (self, new_targ):
        self.targ = np.array (new_targ);
        self.ptarg.set_offsets (self.targ);

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

class Intercept:
    '''
        This class represent and Intercept environment where the goal is to try
        to intercept a moving target by knowledge of its position and current
        velocity. Possible action is the agent move in the next time step.
    '''
    def __init__(self, init = None, targ = None, vtarg = None, dt = 1.,
                    max_T = 500, extent = ((-1, -1), (1, 1)), res = 25, render = True):
        # Here we collect environment duration and span
        self.max_T = max_T;
        self.extent = np.array (extent);
        self.scale = np.mean (self.extent[1] - self.extent[0])
        self.inv_scale = 1 / self.scale;
        self.dt = dt;
        self.res = res;

        # Keep internal time
        self.t = 0;

        # Here we init observation array and done flag
        self.obv = np.empty (2 * res);
        self.done = False;

        # Here we init the position of target and agent
        self.targ = np.array (targ) if targ is not None else np.random.uniform (*extent);
        self.agen = np.array (init) if init is not None else np.zeros (2);

        # Here we init the velocity of the target
        self.vtarg = vtarg if vtarg is not None else np.random.uniform (-1., 1., size = 2);

        self.atraj = np.empty ((2, max_T));
        self.ttraj = np.empty ((2, max_T));

        self.atraj [:, 0] = self.agen;
        self.ttraj [:, 0] = self.targ;

        # Here we init the reward to zero
        self.r = 0.;

        # Here we prepare for rendering
        self.do_render = render;

        if self.do_render:
            self.fig, self.ax = plt.subplots ();
            self.fig.tight_layout ();

            self.ax.set_xlim (self.extent[0][0] + self.agen[0] - 0.1,
                              self.extent[1][0] + self.agen[1] + 0.1);
            self.ax.set_ylim (self.extent[0][1] + self.agen[0] - 0.1,
                              self.extent[1][1] + self.agen[1] + 0.1);

            self.ax.set_xticks (np.arange (-10, 10, 0.5));
            self.ax.set_yticks (np.arange (-10, 10, 0.5));
            self.ax.set_xticklabels ([]);
            self.ax.set_yticklabels ([]);

            # Bbox object around which the fancy box will be drawn.
            # bb = mtransforms.Bbox(self.extent)
            # domain = FancyBboxPatch((bb.xmin, bb.ymin),
            #                         abs(bb.width), abs(bb.height),
            #                         boxstyle="sawtooth,pad=.02",
            #                         fc = 'none',
            #                         ec = 'k',
            #                         zorder = -10);
            #
            # self.ax.add_patch(domain);

            self.ptarg = self.ax.scatter (*self.targ, s = 500, marker = '*', color = 'darkred');
            self.pagen = self.ax.scatter (*self.agen, s = 60, marker = 'p', color = 'darkblue');

            self.pttraj, = self.ax.plot (*self.targ, c = 'C3', ls = '--');
            self.patraj, = self.ax.plot (*self.agen, c = 'C0');

            self.time = self.ax.text (-0.2, 1.1, 'Time 0', fontsize = 16);
            plt.ion();

    def step(self, action):
        # Here we apply the provided action to the agent state and update the target.
        self.agen += np.array (action) * self.dt;
        self.targ += self.vtarg * self.dt;

        self.atraj [:, self.t] = self.agen;
        self.ttraj [:, self.t] = self.targ;

        dist = np.sqrt (np.square (self.targ - self.agen).mean ());

        self.obv [:] = self.encode (self.targ - self.agen);
        self.r = self.dense_r (dist);

        self.done = dist < 0.02 * self.scale or self.t > self.max_T

        # Here we increase the env time
        self.t += 1

        return self.obv, self.r, self.done, self.agen;

    def render(self, cam = 'middle'):
        self.patraj.set_data (*self.atraj[:, :self.t]);
        self.pttraj.set_data (*self.ttraj[:, :self.t]);

        self.pagen.set_offsets (self.atraj[:, self.t - 1]);
        self.ptarg.set_offsets (self.ttraj[:, self.t - 1]);


        if cam == 'agen':
            min_x = self.extent[0][0] + self.atraj[0, self.t - 1] - 0.1;
            max_x = self.extent[1][0] + self.atraj[0, self.t - 1] + 0.1;
            min_y = self.extent[0][1] + self.atraj[1, self.t - 1] - 0.1;
            max_y = self.extent[1][1] + self.atraj[1, self.t - 1] + 0.1;

        elif cam == 'middle':
            min_x = min (self.ttraj[0, self.t - 1], self.atraj[0, self.t - 1]) - 0.5;
            max_x = max (self.ttraj[0, self.t - 1], self.atraj[0, self.t - 1]) + 0.5;
            min_y = min (self.ttraj[1, self.t - 1], self.atraj[1, self.t - 1]) - 0.5;
            max_y = max (self.ttraj[1, self.t - 1], self.atraj[1, self.t - 1]) + 0.5;

        else:
            raise ValueError('Unknwon camera option: {}'.format (cam));

        self.time.set_text ('Time {}'.format (self.t));
        self.time.set_position ((0.5 * (min_x + max_x) - 0.05 * (max_x - min_x),
                                 max_y - 0.1 * (max_y - min_y)));

        self.ax.set_xlim (min_x, max_x);
        self.ax.set_ylim (min_y, max_y);

        # Here we signal redraw
        self.fig.canvas.draw ();
        self.fig.canvas.flush_events();
        plt.show ();
        sleep (0.02);

        return self.fig;

    def dense_r(self, dist):
        # Dense reward is defined as decaying with the distance between agent
        # position and target.
        return np.exp (-dist * self.inv_scale);

    def encode(self, pos, res = None):
        if res is None: res = self.res;

        pos = np.array (pos);
        if len (pos.shape) == 1: pos = pos.reshape (-1, 1);
        shape = pos.shape;

        x, y = np.clip (pos.T, *self.extent).T;

        mu_x, mu_y = np.linspace (*self.extent, num = res).T;
        s_x, s_y = np.diff (self.extent, axis = 0).T / (res);

        enc_x = np.exp (-0.5 * ((x.reshape (-1, 1) - mu_x) / s_x)**2).T;
        enc_y = np.exp (-0.5 * ((y.reshape (-1, 1) - mu_y) / s_y)**2).T;

        return np.array ((enc_x, enc_y)).reshape(-1, shape[-1]).squeeze ();

    def reset_target (self, new_targ = None, new_vtarg = None):
        self.targ = np.array (new_targ) if new_targ is not None else np.random.uniform (-1, 1, size = 2);
        self.vtarg = np.array (new_vtarg) if new_vtarg is not None else np.random.uniform (-1, 1, size = 2);

        self.ptarg.set_offsets (self.targ);

    def reset (self, init = None, targ = None, vtarg = None):
        self.agen = np.array (init) if init is not None else np.random.uniform (-1, 1, size = 2);
        self.targ = np.array (targ) if targ is not None else np.random.uniform (-1, 1, size = 2);
        self.vtarg = np.array (vtarg) if vtarg is not None else np.random.uniform (-1, 1, size = 2);

        self.atraj *= 0.;
        self.ttraj *= 0.;

        self.atraj [:, 0] = self.agen;
        self.ttraj [:, 0] = self.targ;

        self.pagen.set_offsets (self.agen);
        self.ptarg.set_offsets (self.targ);

        self.patraj.set_data (*self.agen);
        self.pttraj.set_data (*self.targ);

        self.t = 0;

    def build_expert (self, targ, init, vtarg, steps = 80, T = 100, offT = 1, norm = True):
        assert T > steps;

        end_targ = targ + vtarg * self.dt * (steps + offT);

        v = np.tile ((end_targ - init) / (steps * self.dt), (steps, 1)).T;
        t = np.linspace (0, steps * self.dt, num = steps);

        inp = init.reshape (2, -1) + v * t;
        out = v;

        inp = np.pad (inp, ((0, 0), (offT, T - offT - inp.shape [-1])), mode = 'edge');
        out = np.pad (out, ((0, 0), (offT, T - offT - out.shape [-1])));

        inp = np.linspace (targ, targ + vtarg * self.dt * T, num = T).T - inp

        return self.encode (inp), out;
