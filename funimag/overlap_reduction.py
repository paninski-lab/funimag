import numpy as np
import scipy as sp
from . import overlap_graph


def difference_operator(len_signal):
    """
    :param len_signal: length of signal
    :return: Difference operator (order=2)
    """
    # Gen Diff matrix
    diff_mat = (np.diag(2 * np.ones(len_signal), 0) +
                np.diag(-1 * np.ones(len_signal - 1), 1) +
                np.diag(-1 * np.ones(len_signal - 1), -1)
                )[1:len_signal - 1]
    return diff_mat


def find_temporal_component(Vt,
                            D,
                            l1tf_th=2.25,
                            sk_thres=2):
    """
    :param Vt: temporal component
    :param D:  difference operator
    :param l1tf_th: threshold to discard components according to
                    TF cte
    :param sk_thres: threshold to discard components according to
                    skeweness
    :return: flag to define if component should be kept
    """
    if np.ndim(Vt) == 1:
        Vt = Vt[np.newaxis, :]

    num_components, L = Vt.shape

    # Include components based on test statistic

    # TF_norm / L1_nom
    D1 = np.sum(np.abs(D.dot(Vt.T)), 0)
    n1 = np.sum(np.abs(Vt), 1)
    keep = np.argwhere((D1/n1) < l1tf_th).flatten()

    # Reject components based on test statistic
    if True:
        zs, ps = sp.stats.skewtest(Vt, axis=1)
        reject = np.where(np.abs(zs) <= sk_thres)[0]
        keep = np.setdiff1d(keep, reject)

    # Plot temporal correlations
    return keep


def component_test(ri, test='l1tf', D=None):
    """
    :param ri: residual from estimated temporal component
    :param test: test statistic employed to discard components
    :param D: second order difference operator
    :return: flag indicating if component passed or failed test statistic
    """
    output = []
    sk_thres = 2
    if test == 'kurto':
        zs, ps = sp.stats.skewtest(ri)
        if np.abs(zs) <= sk_thres:
            output = [0]
    elif test == 'l1tf':
        output = find_temporal_component(ri, D=D)
    return output


def trim_component_group(Uta, Vta, plot_en=False, max_cutoff=0.75, D = None):
    """
    Reduce components via backward selection

    Inputs:
    _______
    Uta             :   spatial components
    Vta             :   temporal components
    max_cutoff      :   float
                        percentage upperbound to cutoff components
    plot_en         :   bool
                        flag to plot


    Outputs:
    ________
    includes projection update

    """
    num_components, L = Vta.shape

    # Calculate norms
    norms1 = np.sqrt(np.sum(Vta ** 2, 1))

    S = np.diag(norms1)

    # exclude zeroed-out components
    exclude = np.where(norms1 == 0)[0]
    # exclude = np.flatnonzero(norms1)

    # Rejected components
    reject = np.zeros(num_components).astype('bool')
    reject[exclude] = True

    # Normalize temporal component
    Vta[norms1 > 0] = Vta[norms1 > 0] / norms1[norms1 > 0, np.newaxis]
    # Vta[!reject] = Vta[!reject]/ norms1[!reject] np.newaxis]

    components = np.arange(num_components)

    # include norm and S to distribute
    norms_ = np.sqrt(np.sum(Uta ** 2, 0))

    # order components according to weighted norm
    sidx_ = np.argsort(norms_)

    # exclude components
    sidx_ = sidx_[~(reject.astype('bool'))]
    components = components[~(reject.astype('bool'))]
    #components = components[not(reject)]

    rss_th = 0.03  # 0.03
    rsqrd_th = 0.05  # 0.15

    # Assert max_cutoff
    if max_cutoff < 1:
        #max_thrown_components = max(1, int(np.floor(len(sidx_) * max_cutoff)))
        max_thrown_components = max(1, np.floor(num_components * max_cutoff))
    else:
        max_thrown_components = len(sidx_)

    reject_raw = 0

    for row, idx in enumerate(sidx_):

        if len(np.flatnonzero(reject)) - reject_raw > max_thrown_components:
            break

        nidx = np.setdiff1d(components, idx)
        nidx = np.setdiff1d(nidx, np.flatnonzero(reject))

        vi = Vta[idx, :]
        vni = Vta[nidx, :]

        # Pvj = projection_matrix(vni,)
        # ai = vi.dot(Pvj)
        XXt = vni.dot(vni.T)
        ai = vi.dot(vni.T.dot(np.linalg.inv(XXt)))
        v_hat = ai.dot(vni)

        # Calculate error
        ri = vi - v_hat
        rss = np.sum(ri ** 2)

        residual_component = component_test(ri, D=D, test='l1tf')
        if len(residual_component) > 0 and rss >= rss_th:
            continue

        # -- Reject component
        reject[idx] = 1

        # Distribute spatial energy
        norm_idx = norms1[idx]
        for idx_nidx, cidx in enumerate(nidx):
            tmp = norm_idx * ai[idx_nidx] / norms1[cidx]
            Uta[:, cidx] += Uta[:, idx] * tmp
        # Delete from memory
        Uta[:, idx] = 0
        Vta[idx, :] = 0


    Vta = S.dot(Vta)

    return Uta, Vta, reject



def rx_graph_adjacent_neighbors(nblocks):
    """
    """
    max_single = int(np.prod(nblocks))

    nng2 = overlap_graph.rx_graph(nblocks)
    Cliques = []
    for tile in range(max_single):
        neigh_tile_idx = np.nonzero(nng2[tile, :])[0]
        if len(neigh_tile_idx) == 0:
            continue

        Clique = np.sort(np.insert(neigh_tile_idx, 0, tile))
        Cliques.append(Clique)

    return Cliques


def one_rank_update(B, idx):
    """

    :param B:
    :param idx:
    :return:
    """
    # find the position in new matrix X
    # B = cvinv(Vta)
    ncomp = B.shape[0]
    # pos = find(col==posX0);
    # permute to bring the column at the end in X
    # B = B([1:pos-1 pos+1:end pos],:);
    # B = B(:,[1:pos-1 pos+1:end pos]);
    mask = np.arange(ncomp)
    mask = np.append(np.setdiff1d(mask, idx), idx)
    B = B[mask, :]
    B = B[:, mask]
    # update the inverse by removing the last column
    # F11inv =B(1:end-1,1:end-1);
    # F22inv = B(end,end);
    # u3 = -B(1:end-1,end);
    # u2 = u3/F22inv;
    # Bnew = F11inv - u2*u2'*F22inv;
    F11inv = B[:-1, :-1]
    F22inv = B[-1, -1]
    u3 = -B[:-1, -1][:, np.newaxis]
    u2 = u3 / F22inv
    Bnew = F11inv - F22inv * u2.dot(u2.T)
    return Bnew



def compression_rdx(nblocks, Ur, V, block_ranks):
    """
    :param nblocks:
    :param Ur:
    :param V:
    :param block_ranks:
    :return:
    """
    neigh_groups = rx_graph_adjacent_neighbors(nblocks)
    rank_tiles = overlap_graph.rank_tile(block_ranks, nblocks)

    # Set components
    D = difference_operator(V.shape[1])

    num_rejected = len(neigh_groups)


    for ii, neigh_tiles in enumerate(neigh_groups):
        twin_components, _ = overlap_graph.neigh_components(neigh_tiles, rank_tiles)

        # stack components
        Ustack = Ur[:, twin_components]
        Vstack = V[twin_components, :]

        trim_init = len(np.where(np.all(Vstack == 0, 1))[0])

        if len(twin_components) - trim_init <= 2:
            continue

        Ustack2, Vstack2, _ = trim_component_group(Ustack, Vstack, D=D)

        if np.any(np.isnan(Ustack2)):
            print('There are nan')
            break

        # count empty components
        trim_out = np.where(np.all(Vstack2 == 0, 1))[0]
        num_rejected = len(trim_out) - trim_init
        print('Tile %d from %d' % (ii, len(neigh_groups)))
        print('Rejected %d from %d' % (len(trim_out) - trim_init, len(twin_components)))

        # Update components
        Ur[:, twin_components] = Ustack2
        V[twin_components, :] = Vstack2

    return Ur, V, num_rejected
