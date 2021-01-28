import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.gridspec import GridSpec

def linear_test_plot (load = 'Unlock.pkl', save = 'Linear_Unlock.png'):
    with open (load, 'rb') as f:
        data = pickle.load (f);

    hist, R = data;
    size = np.size (R);
    btns = np.linspace (-0.3, 0.3, num = size)

    fig = plt.figure ();
    gs = GridSpec (nrows = 2, ncols = 1, height_ratios = [0.7, 0.3], figure = fig);


    ax1 = fig.add_subplot (gs[0]);
    ax2 = fig.add_subplot (gs[1]);

    cw = cmap.coolwarm;
    yg = cmap.PiYG

    for trj, bx,  c in zip (hist['agent'], btns, np.linspace (0., 1., num = size)):
        ax1.plot (*trj.T, c = cw(c));
        ax1.scatter (bx, 0.5, c = [yg(c)], s = 5);

    ax1.scatter ([0.], [1.], s = 50, marker = '*', c = 'C3');

    # ax1.set_xlim ([-0.3, 0.3]);Ù
    ax1.set_ylim ([-0.05, 1.05]);

    ax2.errorbar (btns, R, ls = '--', ms = 5, fmt = 'o', mfc = 'C3', mec = 'w');

    ax2.set_xlabel ('Btn x');
    ax2.set_ylabel ('d (agent, targ)');
    ax2.set_ylim (-0.05, .5);
    ax2.set_yticks (ax2.get_yticks());

    ax2.spines['top'].set_visible (False);
    ax2.spines['right'].set_visible (False);
    ax2.spines['bottom'].set_bounds (*btns[[0, -1]])
    ax2.spines['left'].set_bounds (min (ax2.get_yticks()), max (ax2.get_yticks ()))

    fig.savefig (save);
    plt.show ();

def round_test_plot (load = 'Low_Rank.pkl', save = 'Low_Rank.png'):
    with open (load, 'rb') as f:
        data = pickle.load (f);

    train_T, ranks_data = data;

    Rs = np.array ([rank[2] for rank in ranks_data]);
    Rm = np.mean (Rs, axis = 1);

    best_id = np.argmin (Rm);

    fig = plt.figure (figsize = (10, 8));

    ax1 = fig.add_axes ([0.02, 0.4, 0.3, 0.4]);
    ax2 = fig.add_axes ([0.35, 0.5, 0.3, 0.4]);
    ax3 = fig.add_axes ([0.68, 0.4, 0.3, 0.4]);

    ax4 = fig.add_axes ([0.07, 0.07, 0.9, 0.3]);

    # cw = cmap.twilight_shifted;
    cw = cmap.coolwarm;
    yg = cmap.PiYG


    def _plot (ax, data):
        (Theta, hist, R) = data

        size = np.size (R);
        C = np.linspace (0., 1., num = size);

        rt, rb = .7, -.0;
        # Theta = np.linspace (30., 70, num = size) * np.pi / 180.;
        btns = np.array ([(0., rb)] * size);
        tars = np.array ([(rt * np.cos (t), rt * np.sin (t)) for t in Theta]);
        ang_pos = np.array ([(np.cos (t), np.sin (t)) for t in Theta]);

        for trj, (tx, ty), (bx, by), r,  c in zip (hist['agent'], tars, btns, R, C):
            ax.plot (*(trj.T), c = cw(c), alpha = 1. if r < .1 else 0.1);
            ax.scatter (tx, ty, s = 30, marker = '*', c = [cw(c)], alpha = 1. if r < .1 else .1);
            # ax1.scatter (bx, by, c = [cw(c)], s = 5, alpha = 1. if r < .5 else 0.1);

        ax.scatter (*btns[0], c = 'g', s = 5);
        [ax.plot ([0, (rt + .05) * np.cos(t)], [0, (rt + .05) * np.sin(t)], ls = '--', c = 'forestgreen') for t in train_T]

        xmin = np.min (tars[:, 0]);
        xmax = np.max (tars[:, 0]);
        ymax = np.max (tars[:, 1]);

        ax.set_xlim ([xmin - 0.05, xmax + 0.05]);
        ax.set_ylim ([-0.05, ymax + 0.05]);

    _plot (ax1, ranks_data[0]);
    _plot (ax2, ranks_data[best_id]);
    _plot (ax3, ranks_data[-1]);

    ax1.set_title (f'Rank = {50}');
    ax2.set_title (f'Rank = {50 + best_id}');
    ax3.set_title (f'Rank = {100}');

    [ax.axis ('off') for ax in (ax1, ax2, ax3)];

    ranks = np.arange (50, 101, 1);
    ax4.errorbar (ranks, Rm, fmt = 'o', ls = '--', mfc = 'r');

    ax4.arrow (x = 50 + best_id, y = .18, dx = 0., dy = -0.05, head_length = 0.02,
                width = 0.12, color = 'k');

    ax4.set_xlabel ('rank');
    ax4.set_ylabel (r'$\langle d_{\mathrm{end}} (p, t) \rangle$');

    ax4.spines['top'].set_visible (False);
    ax4.spines['right'].set_visible (False);
    ax4.spines['left'].set_bounds (.1, .3);
    ax4.spines['bottom'].set_bounds (50, 100);

    fig.savefig (save);
    plt.show ();

round_test_plot ();
