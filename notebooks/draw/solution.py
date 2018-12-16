

class Candidate:
    path = None     # Ordered nodes
    nodes = None    # All nodes
    pos = None      # Current position to look

    def __init__(self):
        self.path = []
        self.nodes = set([])
        self.pos = 0


# Data matrix
X = np.array([
       #AJA #ATM  #LIV  #LYO  #MUN  #ROM  #SCH  #TOT
       [1,  0,    1,    1,    1,    1,    1,    0], #BAR
       [0,  1,    1,    1,    1,    1,    0,    1], #BAY
       [1,  0,    1,    1,    1,    1,    0,    1], #BOR
       [1,  1,    1,    1,    0,    0,    1,    1], #JUV
       [1,  1,    0,    0,    0,    1,    1,    0], #MCI
       [1,  1,    0,    0,    1,    1,    1,    1], #PSG
       [1,  1,    1,    1,    1,    1,    0,    1], #POR
       [1,  0,    1,    1,    1,    0,    1,    1]  #REA
    ])

# Name rows and columns for show
rownames = ["Barcelona", "Bayern M.", "Borussia D.", "Juventus", "Manchester C.", "Paris S.G.", "Porto", "Real M."]
colnames = ["Ajax", "Atletico M.", "Liverpool", "Lyon", "Manchester U.", "Roma", "Schalke 04", "Tottenham H."]


def matrix2sets(X):
    """ Convert a data matrix to list of sets. """
    n = X.shape[0]
    data = [set({}) for i in range(n)]
    for r, c in zip(*np.where(X)):
        data[r].add(c)
    return data


def hm(data, rownames, colnames, f_out=None):
    """ Heatmap annotated with text."""
    heatmap = plt.pcolor(data, vmin=0, vmax=66)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.1f' % data[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',)
    cb = plt.colorbar(heatmap)
    cb.set_label("Probability (%)", horizontalalignment='center')
    n = len(data)
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.gca().set_xticks(np.linspace(0.5, n-0.5, n))
    plt.gca().set_yticks(np.linspace(0.5, n-0.5, n))
    plt.gca().set_xticklabels(colnames, rotation=90)
    plt.gca().set_yticklabels(rownames)
    plt.xlim(0, n)
    plt.ylim(0, n)
    if f_out is None:
        plt.show()
    else:
        plt.savefig(f_out, bbox_inches="tight")
        plt.close()
        print("Written %s" % f_out)

# Solution with tail recursion
def process(data):
    """ Return a count matrix of how often each pair appears in the path. """

    # Add an empty path to stack
    stack = [Candidate()]
    n = len(data)

    # Counter matrix
    C = np.zeros((len(data), len(data)))

    # Keep extending paths until hitting end or
    #  no options to extend
    while not len(stack) == 0:
        curr = stack.pop()
        if curr.pos == n:
            C[(list(range(n)),
               curr.path)] += 1
            continue

        # Create new candidate paths
        nodes = data[curr.pos] - curr.nodes
        for no in nodes:
            new = Candidate()
            new.path = curr.path + [no]
            new.nodes = curr.nodes | {no}
            new.pos = curr.pos + 1
            stack.append(new)

    return C






