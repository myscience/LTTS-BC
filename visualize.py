import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def style_ax (ax, lim):
    if lim[0]: ax.set_xlim (*lim[0]);
    if lim[1]: ax.set_ylim (*lim[1]);
    ax.set_xticks (ax.get_xticks ());
    ax.set_yticks (ax.get_yticks ());

    ax.spines['top'].set_visible (False);
    ax.spines['right'].set_visible (False);
    ax.spines['left'].set_bounds (*ax.get_ylim());
    ax.spines['bottom'].set_bounds (*ax.get_xlim());

def cloning_plot (Targ, Agent, save = None):
    targ, out = Targ;
    behv, act = Agent;

    fig = plt.figure ();
    gs = GridSpec (nrows = 2, ncols = 2, figure = fig);

    ax1 = fig.add_subplot (gs[0, 0]);
    ax2 = fig.add_subplot (gs[0, 1]);
    ax3 = fig.add_subplot (gs[1, :]);

    ax1.imshow (targ, aspect = 'auto', cmap = 'binary');
    ax2.imshow (behv, aspect = 'auto', cmap = 'binary');

    ax3.plot (out.T);
    ax3.plot (act.T);

    style_ax (ax1, ((0, 100), (0, 100)))
    style_ax (ax2, ((0, 100), (0, 100)))
    style_ax (ax3, ((0, 100), None))

    fig.text (0., 0., 'Cloning Err: {}'.format (np.abs (targ - behv).sum ()))

    if save:
        fig.savefig (save);
        plt.close (fig);
    else: plt.show (fig);

def env_hist_plot (hist, save = None):
    T = hist['T'];

    fig = plt.figure (figsize = (9, 10));
    gs = GridSpec (nrows = 4, ncols = 2, figure = fig);

    ax1 = fig.add_subplot (gs[:2, 0]);
    ax2 = fig.add_subplot (gs[0, 1]);
    ax3 = fig.add_subplot (gs[1, 1]);
    ax4 = fig.add_subplot (gs[2, 0]);
    ax5 = fig.add_subplot (gs[2, 1]);
    ax6 = fig.add_subplot (gs[3, :]);

    # Visualize env tranjectory
    ax1.plot (*hist['agent'][:, :T]);
    ax1.plot (*hist['targ'][:, :T], c = 'C3', ls = '--');
    ax1.scatter (*hist['agent'][:, T], color = 'b', marker = 'p', s = 50);
    ax1.scatter (*hist['targ'][:, T], color = 'firebrick', marker = '*', s = 50);

    ax1.set_xticks ([]);
    ax1.set_yticks ([]);
    ax1.set_xlim (np.min ([hist['agent'][0, :T], hist['targ'][0, :T]]) - 0.5,
                  np.max ([hist['agent'][0, :T], hist['targ'][0, :T]]) + 0.5);
    ax1.set_ylim (np.min ([hist['agent'][1, :T], hist['targ'][1, :T]]) - 0.5,
                  np.max ([hist['agent'][1, :T], hist['targ'][1, :T]]) + 0.5);

    # Visualize trajectory components and errors
    ax2.plot (hist['agent'][0, :T], c = 'darkred');
    ax2.plot (hist['agent'][1, :T], c = 'darkblue');
    ax2.set_ylabel ('trajectory');

    ax3.plot (hist['targ'][0, :T] - hist['agent'][0, :T], 'darkgreen');
    ax3.plot (hist['targ'][1, :T] - hist['agent'][1, :T], 'gold');
    ax3.set_ylabel ('error');

    # Visualize input to agent and output to environment
    ax4.plot (hist['obv'][0, :T], c = 'r', label = 'observation');
    ax4.plot (hist['obv'][1, :T], c = 'g', label = 'observation');
    ax4.set_ylabel ('observation');

    ax5.plot (hist['act'][0, :T], c = 'C1', label = 'actions');
    ax5.plot (hist['act'][1, :T], c = 'firebrick', label = 'actions');
    ax5.set_ylabel ('action');

    # Visualize network spiking dynamics
    ax6.imshow (hist['S'][:, :T], aspect = 'auto', cmap = 'binary');
    ax6.set_ylabel ('# neurons');
    ax6.set_xlabel ('time');

    for ax in (ax2, ax3, ax4, ax5):
        style_ax (ax, (None, None));

    fig.tight_layout ();

    if save:
        fig.savefig (save);
        plt.close (fig);
    else: plt.show (fig);
