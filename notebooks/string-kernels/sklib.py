import numpy as np
import scipy.sparse as sp
import itertools as it
from scipy.special import binom
from collections import Counter
from copy import deepcopy

hlp = """ Compute kernels between lists of sequences with sparse matrices. """


### Explicit formulation of kernels

def miss(s, t):
    """ Count the number of mismatches between two strings."""
    return sum((si != sj for si, sj in zip(s, t)))

def string_kernel(sx, sy, k=4, delta=2):
    """ Basic string kernel with displacement. """
    L = len(sx)
    return sum(((sx[i:i + k] == sy[d + i:d + i + k])
                for i, d in it.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0))

def string_kernel_mismatch(s, t, k=4, delta=2, m=1):
    """ Basic string kernel with displacement. """
    L = len(s)
    return sum(((miss(s[i:i + k], t[d + i:d + i + k]) <= m)
                for i, d in it.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0))

def string_kernel_mismatch_exp(s, t, k=3, delta=1, m=1, gamma=0.99):
    """ String kernel with displacement, mismatches and exponential decay. """
    L = len(s)
    return sum(((np.exp(-gamma * d**2)
                 * np.exp(-gamma * miss(s[i:i + k], t[d + i:d + i + k]))
                 * (miss(s[i:i + k], t[d + i:d + i + k]) <= m)
                for i, d in it.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0)))

def kernel_matrix(S, kernel, **kwargs):
    """ Compute the kernel matrix between all pairs of strings in a list S"""
    N = len(S)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = kernel(S[i], S[i], **kwargs)
    for i, j in it.combinations(range(N), 2):
        K[i, j] = K[j, i] = kernel(S[i], S[j], **kwargs)
    return K


### Sparse matrix implementations

# Caching of kernels
_coef_cache = dict()


# General wrapper
def string_kernel_matrix(X1, X2, **kwargs):
    """
    :param X1: Strings represented as a sparse k-mer matrix.
    :param X2: Strings represented as a sparse k-mer matrix.
    :param kwargs: Arguments to the kernel ().
    :return:
    """
    global _coef_cache
    mode = frozenset(kwargs.items())
    ky = hash(mode)
    if ky not in _coef_cache:
        K = _coef_cache[ky] = coefficients(**kwargs)
    else:
        K = _coef_cache[ky]
    return X1.dot(K).dot(X2.T)


def coefficients(L, k, m=0, delta=1, gamma=1, base=4):
    """
    Transform related to string kernels. Various scaling functions are possible given a delta.
    :param L: Sequence length.
    :param k: K-mer length.
    :param m: Allowed number of mismatches.
    :param delta: Offset for position-invariant string kernel.
    :param gamma: Bandwidth parameter for the kernel.
    :param base: Alphabet size.
    :return:
    """
    M = mismatch_matrix(k=k, m=m, gamma=gamma, base=base)
    D = distance_matrix(L=L, k=k, delta=delta, gamma=gamma)
    return sp.kron(D, M, format="csr")


def mismatch_matrix(k, m, base=4, gamma=1):
    """
    Compute a (sparse) matrix of k-mers different in at most m places.
    Sums are modulo base.
    :param k: k-mer length.
    :param m: Number of mismatches (m < k)
    :param base: Alphabet size.
    :param gamma: Bandwidth (damping) factor.
    :return:
    """
    assert m < k
    a = base ** k
    if m == 0:
        return sp.eye(a, a, dtype=float, format="csr")
    M = sp.lil_matrix((a, a), dtype=float)
    nvec = sum((int(binom(k, mi)) * (base-1) ** mi for mi in range(m+1)))
    inc = np.zeros((nvec, k), dtype=float)   # Increment vectors
    D = np.zeros((nvec,), dtype=float)     # Distances
    count = 1
    for mi in range(1, m+1):
        P = np.array(list(it.product(*(mi*[range(1, base)]))))
        for cols in it.combinations(range(k), mi):
            inc[count:count+P.shape[0], list(cols)] = P
            D[count:count+P.shape[0]] = mi
            count += P.shape[0]
    basis = np.power(base, np.arange(k)[::-1])
    vecs = np.array(list(it.product(*(k*[range(base)]))))
    for vec in vecs:
        i = vec.dot(basis)
        inxs = np.mod(inc + vec, base).dot(basis)
        M[i, inxs] = np.exp(-gamma * D)
    return M.tocsr()


def distance_matrix(L, k, delta=1, gamma=1):
    """
    Generate a matrix that decreases with k-mer distance.
    :param L: Sequence length.
    :param k: K-mer length (required to generate possible distances).
    :param delta: Maximum offsets.
    :param gamma: Bandwidth (damping) parameter for the kernel.
    :return:
    """
    d = (L - k + 1)
    D = np.ones((2 * delta + 1, d), dtype=float)
    offsets = np.arange(-delta, delta+1)
    facs = np.exp(-gamma * np.arange(-delta, delta+1)**2).reshape((2*delta+1, 1))
    return sp.spdiags(facs * D, offsets, d, d)


def sequence_padding(seq_list, fill="N", align="start"):
    """
    Ensure sequences are of the same length by padding with additional characters.
    :param seq_list: Iterable of sequences, to be replace in place.
    :param fill: replacement character.
    :param align: 'start', 'end' or 'center'.
    :return: Padded sequences.
    """
    seq_return = deepcopy(seq_list)
    max_len = max(map(len, seq_list))
    for i in range(len(seq_list)):
        s = seq_list[i]
        r = max_len - len(s)
        if align == "center":
            start = int(np.floor(r / 2.0))
            end = int(np.ceil(r / 2.0))
        elif align == "start":
            start = 0
            end = r
        else:
            start = r
            end = 0
        seq_return[i] = start * fill + s + end * fill
    return seq_return


def kmer_sparse_matrix(seq_list, k, mode="spectrum", alphabet=("A", "C", "G", "T")):
    """
    Generate a k-mer sparse matrix.
    :param seq_list: List of sequences.
    :param k: K-mer length.
    :param mode: 'positional' or 'spectrum'.
    :param alphabet: Expected character alphabet.
    :return: Sparse matrix and a list of columns.
    """
    # Count records
    lengths = set(list(map(len, seq_list)))
    if len(lengths) != 1:
        raise ValueError("Sequences are not of the same length (%d)!" % len(lengths))

    # Aux functions
    tup2kmer = lambda km: "".join(km)
    seq2tups = lambda sq: zip(*[sq[ki:] for ki in range(k)])

    # Generate kmer matrix from unambiguous DNA
    kmers = set(map(tup2kmer, it.product(*[alphabet] * k)))
    if mode == "positional":
        L = lengths.pop() - k + 1
        cols = list(map(lambda t: "%s_%d" % (t[1], t[0]), it.product(range(L), sorted(kmers))))
    elif mode == "spectrum":
        cols = kmers
    else:
        raise ValueError("Unknown mode: %s" % mode)

    # Initialize matrix
    col_dict = dict(((c, i) for i, c in enumerate(cols)))
    X = sp.lil_matrix((len(seq_list), len(cols)))

    # Fill matrix
    for i, record in enumerate(seq_list):
        seq = str(record).upper()
        if mode == "positional":
            sc = ["%s_%d" % (ky, j) for j, ky in enumerate(map(tup2kmer, seq2tups(seq)))]
            inxs = np.fromiter(filter(lambda j: j is not None,
                                      map(lambda t: col_dict.get(t, None), sc)),
                               dtype=int)
            X[i, inxs] = 1
        elif mode == "spectrum":
            row = Counter(map(tup2kmer, seq2tups(seq)))
            tups = filter(lambda tup: tup[1] is not None,
                          map(lambda kmer: (kmer, col_dict.get(kmer, None)), row.iterkeys()))
            inxs, vals = zip(*map(lambda tup: (tup[1], row[tup[0]]), tups))
            X[i, inxs] = vals

    return X.tocsr(), cols



