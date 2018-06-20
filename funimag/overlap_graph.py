import numpy as np

# To do
# Update graph representation

def component_idx(tile_rank, idx):
    """
    Inputs:
    _______
    tile_rank       :   int

    Outputs:
    _______
    range           :   np.array (L,)
                        Vector with
    """
    init = tile_rank[:idx].sum()
    length = tile_rank[idx]
    return np.arange(init, init + length, dtype='int')


def neigh_components(neigh_tiles, rank_tiles):
    """
    Visualization

    Inputs:
    _______
    Outputs:
    _______
    """
    twin_components = []
    breaks = [0]
    for idx in neigh_tiles:
        twin_components = np.concatenate([twin_components,
                                          component_idx(rank_tiles, idx)])
        # print(component_idx(rank_tiles,idx))
        breaks.append(component_idx(rank_tiles, idx).shape[0])
        # print(component_idx(rank_tiles,idx).shape[0])
    return twin_components.astype('int'), breaks


def rank_tile(block_ranks, nblocks):
    """
    Extract rank for a given input
    Inputs:
    _______
    block_ranks :
    nblocks     :
    Outputs:
    _______
    rank_tiles  :
    """
    # def overlapping_rank_reformat(block_ranks,nblocks):
    d1, d2 = nblocks[:2]
    num_no_offset = int(d1*d2)
    num_row_offset = int((d1-1)*d2)
    num_col_offset = int(d1*(d2-1))
    num_diag_offset = int((d1-1)*(d2-1))

    num_row_half_tiles = 2*d2
    num_col_half_tiles = 2*d1
    num_diag_rhalf_tiles = d1*2-2
    num_diag_chalf_tiles = d2*2-2
    num_diag_quarter_tiles = 4

    num_total_tiles = num_no_offset + num_row_offset
    num_total_tiles += num_diag_offset + num_row_half_tiles
    num_total_tiles += num_col_offset + num_col_half_tiles
    num_total_tiles += num_diag_rhalf_tiles + num_diag_chalf_tiles
    num_total_tiles += num_diag_quarter_tiles

    rank_tiles = np.zeros((num_total_tiles,))

    cumsum = 0

    for tile_count, tile_name in zip([num_no_offset,
                                      num_row_offset,
                                      num_row_half_tiles,
                                      num_col_offset,
                                      num_col_half_tiles,
                                      num_diag_offset,
                                      num_diag_rhalf_tiles,
                                      num_diag_chalf_tiles,
                                      num_diag_quarter_tiles],
                                     [('no_skew', 'full'),
                                      ('vert_skew', 'full'),
                                      ('vert_skew', 'half'),
                                      ('horz_skew', 'full'),
                                      ('horz_skew', 'half'),
                                      ('diag_skew', 'full'),
                                      ('diag_skew', 'thalf'),
                                      ('diag_skew', 'whalf'),
                                      ('diag_skew', 'quarter'),
                                      ]):


        # -- update half skew where components are added in different order
        tmp_rank = block_ranks[tile_name[0]][tile_name[1]]
        if tile_name[1] == 'half' and tile_name[0] == 'vert_skew':
            rank_tiles[cumsum:cumsum + tile_count//2] = tmp_rank[::2]
            rank_tiles[cumsum + tile_count//2: cumsum + tile_count] = tmp_rank[1::2]
        # -- update not half skew where components are added in some order
        elif tile_name[1] == 'whalf':
            rank_tiles[cumsum:cumsum + tile_count//2] = tmp_rank[::2]
            rank_tiles[cumsum + tile_count//2: cumsum +
                       tile_count] = tmp_rank[1::2]
        else:
            rank_tiles[cumsum:cumsum + tile_count] = tmp_rank

        cumsum += len(tmp_rank)# tile_count

    return rank_tiles


def cx_blocks(nblocks):
    """
    Column major ordering for full offset matrices
    Row major ordering for half offset matrices

    Build test cases from which to create connectivity graph
    Inputs:
    ________
    nblocks     :   (d1,d2) tuple
                    indicated the dimensions over which a block was partitioned
    Outputs:
    ________
    M           :   (d1,d2) numpy array
                    indices of tiles for no skew matrix
    Mr          :   (d1,d2) numpy array
                    indices of tiles for vert_skew full matrix
    Mrh         :   (d1,d2) numpy array
                    indices of tiles for vert_skew half matrix
    Mc          :   (d1,d2) numpy array
                    indices of tiles for vert_skew matrix
    Mch         :   (d1,d2) numpy array
                    indices of tiles for vert_skew half matrix
    Md          :   (d1,d2) numpy array
                    indices of tiles for diag_skew full matrix
    Mdr         :   (d1,d2) numpy array
                    indices of tiles for diag_skew thalf matrix
    Mdc         :   (d1,d2) numpy array
                    indices of tiles for diag_skew whalf matrix
    Mdh         :   (d1,d2) numpy array
                    indices of tiles for diag_skew quarter matrix
    """
    d1, d2 = nblocks[:2]
    base = np.arange(1, d1*d2+1).reshape(d2, d1).T
    base_r = np.arange(1, d2*(d1-1)+1).reshape(d2, d1-1).T
    base_c = np.arange(1, d1*(d2-1)+1).reshape(d2-1, d1).T
    base_rc = np.arange(1, (d1-1)*(d2-1)+1).reshape(d2-1, d1-1).T

    M = np.repeat(np.repeat(base, 2, axis=1), 2, axis=0)
    Mr = np.zeros(M.shape)
    Mc = np.zeros(M.shape)
    Md = np.zeros(M.shape)

    M = np.repeat(np.repeat(base, 2, axis=1), 2, axis=0)
    Mr = np.zeros(M.shape)
    Mc = np.zeros(M.shape)
    Md = np.zeros(M.shape)

    Mr[1:-1, :] = np.repeat(np.repeat(base_r, 2, axis=1), 2, axis=0)
    Mc[:, 1:-1] = np.repeat(np.repeat(base_c, 2, axis=1), 2, axis=0)
    Md[1:-1, 1:-1] = np.repeat(np.repeat(base_rc, 2, axis=1), 2, axis=0)

    # --- overcomplete tiles in row order
    Mrh = np.zeros(M.shape)
    Mch = np.zeros(M.shape)
    Mdr = np.zeros(M.shape)
    Mdc = np.zeros(M.shape)
    Mdh = np.zeros(M.shape)

    Mrh[0, :] = np.repeat(np.arange(1, d2+1), 2, axis=0)
    Mrh[-1, :] = np.repeat(np.arange(d2+1, d2*2+1), 2, axis=0)
    Mch[:, 0] = np.repeat(np.arange(1, d1+1), 2, axis=0)
    Mch[:, -1] = np.repeat(np.arange(d1+1, d1*2+1), 2, axis=0)

    # Mdr[0, 1:-1] = np.repeat(np.arange(1, d2*2-1)[::2], 2, axis=0)
    # Mdr[-1, 1:-1] = np.repeat(np.arange(1, d2*2-1)[1::2], 2, axis=0)

    Mdr[1:-1, 0] = np.repeat(np.arange(1, d1), 2, axis=0)
    Mdr[1:-1, -1] = np.repeat(np.arange(d1, 2*d1-1), 2, axis=0)
    
    # Mdc[1:-1, 0] = np.repeat(np.arange(1, d1*2-1)[::2], 2, axis=0)
    # Mdc[1:-1, -1] = np.repeat(np.arange(1, d1*2-1)[1::2], 2, axis=0)

    Mdc[0, 1:-1] = np.repeat(np.arange(1, d2), 2, axis=0)
    Mdc[-1, 1:-1] = np.repeat(np.arange(d2, 2*d2-1), 2, axis=0)

    Mdh[0, 0] = 1
    Mdh[-1, 0] = 2
    Mdh[0, -1] = 3
    Mdh[-1, -1] = 4

    # --- exclude 0 replace by nan
    Mr[Mr == 0] = np.nan
    Mc[Mc == 0] = np.nan
    Md[Md == 0] = np.nan

    Mrh[Mrh == 0] = np.nan
    Mch[Mch == 0] = np.nan
    Mdr[Mdr == 0] = np.nan
    Mdc[Mdc == 0] = np.nan
    Mdh[Mdh == 0] = np.nan

    # --- python 0 indexing
    M -= 1
    Mr -= 1
    Mc -= 1
    Md -= 1

    Mrh -= 1
    Mch -= 1
    Mdr -= 1
    Mdc -= 1
    Mdh -= 1

    return M, Mr, Mrh, Mc, Mch, Md, Mdr, Mdc, Mdh


def rx_blocks(nblocks):
    """
    Row major ordering
    Inputs:
    ________
    nblocks     :   (d1,d2) tuple
                    indicated the dimensions over which a block was partitioned

    Outputs:
    ________
    M           :   (d1,d2) numpy array
                    indices of tiles for no skew matrix
    Mr          :   (d1,d2) numpy array
                    indices of tiles for vert_skew full matrix
    Mrh         :   (d1,d2) numpy array
                    indices of tiles for vert_skew half matrix
    Mc          :   (d1,d2) numpy array
                    indices of tiles for vert_skew matrix
    Mch         :   (d1,d2) numpy array
                    indices of tiles for vert_skew half matrix
    Md          :   (d1,d2) numpy array
                    indices of tiles for diag_skew full matrix
    Mdr         :   (d1,d2) numpy array
                    indices of tiles for diag_skew thalf matrix
    Mdc         :   (d1,d2) numpy array
                    indices of tiles for diag_skew whalf matrix
    Mdh         :   (d1,d2) numpy array
                    indices of tiles for diag_skew quarter matrix
    """
    d1, d2 = nblocks[:2]
    base = np.arange(1, d1*d2+1).reshape(d1, d2)
    base_c = np.arange(1, d1*(d2-1)+1).reshape(d1, d2-1)
    base_rc = np.arange(1, (d1-1)*(d2-1)+1).reshape(d1-1, d2-1)

    M = np.repeat(np.repeat(base, 2, axis=1), 2, axis=0)
    Mr = np.zeros(M.shape)
    Mc = np.zeros(M.shape)
    Md = np.zeros(M.shape)
    Mr[1:-1, :] = np.repeat(np.repeat(base[:d1-1], 2, axis=1), 2, axis=0)
    Mc[:, 1:-1] = np.repeat(np.repeat(base_c, 2, axis=1), 2, axis=0)
    Md[1:-1, 1:-1] = np.repeat(np.repeat(base_rc, 2, axis=1), 2, axis=0)

    Mrh = np.zeros(M.shape)
    Mch = np.zeros(M.shape)
    Mdr = np.zeros(M.shape)
    Mdc = np.zeros(M.shape)
    Mdh = np.zeros(M.shape)

    Mrh[0, :] = np.repeat(np.arange(1, d2+1), 2, axis=0)
    Mrh[-1, :] = np.repeat(np.arange(d2+1, d2*2+1), 2, axis=0)
    Mch[:, 0] = np.repeat(np.arange(1, d1+1), 2, axis=0)
    Mch[:, -1] = np.repeat(np.arange(d1+1, d1*2+1), 2, axis=0)

    # Mdr[0, 1:-1] = np.repeat(np.arange(1, d2*2-1)[::2], 2, axis=0)
    # Mdr[-1, 1:-1] = np.repeat(np.arange(1, d2*2-1)[1::2], 2, axis=0)
    Mdr[1:-1,0] = np.repeat(np.arange(1, d1), 2, axis=0)
    Mdr[1:-1,-1] = np.repeat(np.arange(d1, 2*d1-1), 2, axis=0)

    # Mdc[1:-1, 0] = np.repeat(np.arange(1, d1*2-1)[::2], 2, axis=0)
    # Mdc[1:-1, -1] = np.repeat(np.arange(1, d1*2-1)[1::2], 2, axis=0)
    Mdc[0,1:-1] = np.repeat(np.arange(1, d2), 2, axis=0)
    Mdc[-1,1:-1] = np.repeat(np.arange(1, 2*d2-1), 2, axis=0)

    Mdh[0, 0] = 1
    Mdh[-1, 0] = 2
    Mdh[0, -1] = 3
    Mdh[-1, -1] = 4

    # --- exclude 0 replace by nan
    Mr[Mr == 0] = np.nan
    Mc[Mc == 0] = np.nan
    Md[Md == 0] = np.nan

    Mrh[Mrh == 0] = np.nan
    Mch[Mch == 0] = np.nan
    Mdr[Mdr == 0] = np.nan
    Mdc[Mdc == 0] = np.nan
    Mdh[Mdh == 0] = np.nan

    # --- python 0 indexing
    M -= 1
    Mr -= 1
    Mc -= 1
    Md -= 1

    Mrh -= 1
    Mch -= 1
    Mdr -= 1
    Mdc -= 1
    Mdh -= 1

    return M, Mr, Mrh, Mc, Mch, Md, Mdr, Mdc, Mdh


def rx_neigh(dims, M, Mr):
    """
    Inputs:
    _______
    dims    :
    M       :
    Mr      :

    Outputs:
    ________
    A       :

    """
    d1, d2 = dims[:2]
    A = np.zeros((d1, d2))
    for ii in range(d1):
        nn = np.unique(M[np.nonzero(Mr == ii)])
        nn = nn[~np.isnan(nn)].astype('int')
        A[ii, nn] = 1
    return A


def rx_graph(nblocks):
    """
    Builds connectivity matrix to find temporal components per pixel
    using the pixel components

    Inputs:
    _______
    nblocks  :

    Outputs:
    ________
    M3       :

    """

    d1, d2 = nblocks[:2]
    num_no_offset = int(d1*d2)
    num_row_offset = int((d1-1)*d2)
    num_col_offset = int(d1*(d2-1))
    num_diag_offset = int((d1-1)*(d2-1))

    num_row_half_tiles = 2*d2
    num_col_half_tiles = 2*d1
    num_diag_rhalf_tiles = d1*2-2
    num_diag_chalf_tiles = d2*2-2
    num_diag_quarter_tiles = 4

    # -- master graph
    M, \
    Mr, Mrh, \
    Mc, Mch, \
    Md, Mdr, Mdc, Mdh = cx_blocks([d1, d2])

    # individual graphs
    A = rx_neigh((num_row_offset, num_no_offset), M, Mr)
    B = rx_neigh((num_col_offset, num_no_offset), M, Mc)
    C = rx_neigh((num_diag_offset, num_no_offset), M, Md)
    D = rx_neigh((num_col_offset, num_row_offset), Mr, Mc)
    E = rx_neigh((num_diag_offset, num_row_offset), Mr, Md)
    F = rx_neigh((num_diag_offset, num_col_offset), Mc, Md)

    # overcomplete graphs
    r1o = rx_neigh((num_row_half_tiles, num_no_offset), M, Mrh)
    c1o = rx_neigh((num_col_half_tiles, num_no_offset), M, Mch)
    d1o = rx_neigh((num_diag_rhalf_tiles, num_no_offset), M, Mdr)
    d2o = rx_neigh((num_diag_chalf_tiles, num_no_offset), M, Mdc)
    d3o = rx_neigh((num_diag_quarter_tiles, num_no_offset), M, Mdh)
    #
    # c1r = rx_neigh((num_col_half_tiles, num_row_offset), Mch, Mch)
    # d1r = rx_neigh((num_diag_rhalf_tiles, num_row_offset), Mch, Mdr)
    # d2r = rx_neigh((num_diag_chalf_tiles, num_row_offset), Mch, Mdc)
    # d3r = rx_neigh((num_diag_quarter_tiles, num_row_offset), Mch, Mdh)

    c1r = rx_neigh((num_col_half_tiles, num_row_offset), Mr, Mch)
    d1r = rx_neigh((num_diag_rhalf_tiles, num_row_offset), Mr, Mdr)
    d2r = rx_neigh((num_diag_chalf_tiles, num_row_offset), Mr, Mdc)
    d3r = rx_neigh((num_diag_quarter_tiles, num_row_offset), Mr, Mdh)

    #
    cr1 = rx_neigh((num_col_offset, num_row_half_tiles), Mrh, Mc)
    c1r1 = rx_neigh((num_col_half_tiles, num_row_half_tiles), Mrh, Mch)
    dr1 = rx_neigh((num_diag_offset, num_row_half_tiles), Mrh, Md)
    d1r1 = rx_neigh((num_diag_rhalf_tiles, num_row_half_tiles), Mrh, Mdr)
    d2r1 = rx_neigh((num_diag_chalf_tiles, num_row_half_tiles), Mrh, Mdc)
    d3r1 = rx_neigh((num_diag_quarter_tiles, num_row_half_tiles), Mrh, Mdh)
    #
    d1c = rx_neigh((num_diag_rhalf_tiles, num_col_offset), Mc, Mdr)
    d2c = rx_neigh((num_diag_chalf_tiles, num_col_offset), Mc, Mdc)
    d3c = rx_neigh((num_diag_quarter_tiles, num_col_offset), Mc, Mdh)
    #
    dc1 = rx_neigh((num_diag_offset, num_col_half_tiles), Mch, Md)
    d1c1 = rx_neigh((num_diag_rhalf_tiles, num_col_half_tiles), Mch, Mdr)
    d2c1 = rx_neigh((num_diag_chalf_tiles, num_col_half_tiles), Mch, Mdc)
    d3c1 = rx_neigh((num_diag_quarter_tiles, num_col_half_tiles), Mch, Mdh)

    # --- merge all blocks into one single array
    #        |  off |  row | row1  | col   | col1   |  diag  | diag1  | diag2 |  diag3  |
    #  off   |  X   |  A.T |  r1o.T| B.T   | c1o.T  |  C.T   | d1o.T  | d2o.T |  d3o.T  |
    #  row   |  A   |  X   |  X    | D.T   | c1r.T  |  E.T   | d1r.T  | d2r.T |  d3r.T  |
    #  row1  |  r1o |  X   |  X    | cr1.T | c1r1.T |  dr1.T | d1r1.T | d2r1.T|  d3r1.T |
    #  col   |  B   |  D   |  cr1  | X     |  X     |  F.T   | d1c.T  | d2c.T |  d3c.T  |
    #  col1  |  c1o |  c1r |  c1r1 | X     |  X     |  dc1.T | d1c1.T | d2c1.T|  d3c1.T |
    #  diag  |  C   |  E   |  dr1  | F     | dc1    |  X     | X      |  X    |   X     |
    #  diag1 |  d1o |  d1r |  d1r1 | d1c   | d1c1   |  X     | X      |  X    |   X     |
    #  diag2 |  d2o |  d2r |  d2r1 | d2c   | d2c1   |  X     | X      |  X    |   X     |
    #  diag3 |  d3o |  d3r |  d3r1 | d3c   | d3c1   |  X     | X      |  X    |   X     |

    # OFF COL
    X = np.zeros((num_no_offset, num_no_offset))
    COL1 = np.vstack([X, A, r1o, B, c1o, C, d1o, d2o, d3o])

    # ROW COL
    X1 = np.zeros((num_row_offset, num_row_offset))
    X2 = np.zeros((num_row_half_tiles, num_row_offset))
    COL2 = np.vstack([A.T, X1, X2, D, c1r, E, d1r, d2r, d3r])

    # ROW1 COL
    X1 = np.zeros((num_row_offset, num_row_half_tiles))
    X2 = np.zeros((num_row_half_tiles, num_row_half_tiles))
    COL3 = np.vstack([r1o.T, X1, X2, cr1, c1r1, dr1, d1r1, d2r1, d3r1])

    # COL COL
    X1 = np.zeros((num_col_offset, num_col_offset))
    X2 = np.zeros((num_col_half_tiles, num_col_offset))
    COL4 = np.vstack((B.T, D.T, cr1.T, X1, X2, F, d1c, d2c, d3c))

    # COL1 COL
    X1 = np.zeros((num_col_offset, num_col_half_tiles))
    X2 = np.zeros((num_col_half_tiles, num_col_half_tiles))
    COL5 = np.vstack([c1o.T, c1r.T, c1r1.T, X1, X2, dc1, d1c1, d2c1, d3c1])

    # DIAG COL
    X1 = np.zeros((num_diag_offset, num_diag_offset))
    X2 = np.zeros((num_diag_rhalf_tiles, num_diag_offset))
    X3 = np.zeros((num_diag_chalf_tiles, num_diag_offset))
    X4 = np.zeros((num_diag_quarter_tiles, num_diag_offset))
    COL6 = np.vstack([C.T, E.T, dr1.T, F.T, dc1.T, X1, X2, X3, X4])

    # DIAG1 COL
    X1 = np.zeros((num_diag_offset, num_diag_rhalf_tiles))
    X2 = np.zeros((num_diag_rhalf_tiles, num_diag_rhalf_tiles))
    X3 = np.zeros((num_diag_chalf_tiles, num_diag_rhalf_tiles))
    X4 = np.zeros((num_diag_quarter_tiles, num_diag_rhalf_tiles))
    COL7 = np.vstack([d1o.T, d1r.T, d1r1.T, d1c.T, d1c1.T, X1, X2, X3, X4])

    # DIAG1 COL
    X1 = np.zeros((num_diag_offset, num_diag_chalf_tiles))
    X2 = np.zeros((num_diag_rhalf_tiles, num_diag_chalf_tiles))
    X3 = np.zeros((num_diag_chalf_tiles, num_diag_chalf_tiles))
    X4 = np.zeros((num_diag_quarter_tiles, num_diag_chalf_tiles))
    COL8 = np.vstack([d2o.T, d2r.T, d2r1.T, d2c.T, d2c1.T, X1, X2, X3, X4])

    # DIAG1 COL
    X1 = np.zeros((num_diag_offset, num_diag_quarter_tiles))
    X2 = np.zeros((num_diag_rhalf_tiles, num_diag_quarter_tiles))
    X3 = np.zeros((num_diag_chalf_tiles, num_diag_quarter_tiles))
    X4 = np.zeros((num_diag_quarter_tiles, num_diag_quarter_tiles))
    COL9 = np.vstack([d3o.T, d3r.T, d3r1.T, d3c.T, d3c1.T, X1, X2, X3, X4])

    M3 = np.hstack([COL1, COL2, COL3, COL4, COL5, COL6, COL7, COL8, COL9])

    return M3
