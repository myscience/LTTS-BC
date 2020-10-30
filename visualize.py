import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import utils as ut

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

    ut.style_ax (ax1, ((0, 100), (0, 100)))
    ut.style_ax (ax2, ((0, 100), (0, 100)))
    ut.style_ax (ax3, ((0, 100), None))

    if save: fig.savefig (save);
    plt.show ();
