import matplotlib.pyplot as plt

import numpy as np
import cv2
import scipy.ndimage.filters as filters

from mpl_toolkits.axes_grid1 import make_axes_locatable


from .. import utils
from .. import superpixel_analysis as sup


# TODO
# update handles st defaults follow matplotlib conventions
# update ticks
# merge calls
# merge functions


def colorbar(mappable,
            cax=None,
            cbar_orientation='horizontal',
            format_tile='%.2f',
            sig_digits = 2,
            cbar_spacing='uniform',
            cbar_ticks=None,
            cbar_ticks_number=5):

    if cbar_orientation == 'horizontal':
        cbar_direction ='bottom'
    elif cbar_orientation == 'vertical':
        cbar_direction ='right'

    if cax is None:
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_direction,
                                size="3%",
                                pad=0.1)
        if cbar_ticks is not None:
            cbar_ticks = np.round(cbar_ticks,4)
            cbar_ticks_labels = [format_tile%(cbar_)
                                for cbar_ in cbar_ticks]
            vmin, vmax = cbar_ticks[0], cbar_ticks[-1]

    cbar = plt.colorbar(mappable,
                        cax=cax,
                        orientation=cbar_orientation,
                        spacing=cbar_spacing,
                        format=format_tile,
                        ticks=cbar_ticks)

    if not cbar_direction == 'bottom':
        cax.yaxis.set_ticks_position(cbar_direction)

    #if np.all([ tick == int(tick) for tick in cbar.get_ticks()]):
    #    cbar.set_ticks(list(map(int, cbar.get_ticks())))

    if cbar_ticks is None:
        cbar_ticks = cbar.get_clim()
    if cbar_ticks_number is not None:
        cbar_ticks=np.linspace(min(cbar_ticks),
                               max(cbar_ticks),
                               cbar_ticks_number)
    
    cbar_ticks = [np.round(tick,
                           sig_digits) for tick in cbar_ticks]

    n_digits = [digits(x) for x in cbar_ticks]

    cbar_ticklabels = [str.format('{0:.'+str(ndig)+'f}',
                                  cdig) for ndig, cdig in zip(n_digits,cbar_ticks)]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    return


def digits(f):
    """
    """
    return len(str(f).split('.')[1].rstrip('0'))


def show_img(img,
             ax=None,
             vmin=None,
             vmax=None,
             interpolation=None,
             title_=None,
             cbar_orientation='horizontal',
             plot_colormap='jet',
             plot_size=(12,7),
             sig_digits=2,
             plot_aspect=None,
             fig_pdf_var=None,
             cbar_direction=None,
             cbar_spacing='uniform',
             cbar_pad=0.1,
             cbar_size="3%",
             cbar_ticks_number=None,
             cbar_ticks=None,
             cbar_enable=True):
    """
    Visualize image
    """
    if ax is None:
        fig = plt.figure(figsize=plot_size)
        ax = plt.subplot(111)

    ax.set_title(title_)

    vmin= img.min() if vmin is None else vmin
    vmax= img.max() if vmax is None else vmax

    if np.abs(img.min()) <= 1.5 or np.abs(img.min()-img.max())<=1.5 :
        if np.abs(img.min()) <= -1e-1:
            format_tile = '%.1e'
        else:
            format_tile = '%.2f'
    else:
        format_tile = '%5d'

    if cbar_ticks_number is not None:
        cbar_ticks= np.linspace(vmin,
                                vmax,
                                cbar_ticks_number,
                                endpoint=True)
        cbar_ticks = np.round(cbar_ticks,sig_digits)
        cbar_ticks_labels = [format_tile%(cbar_)
                            for cbar_ in cbar_ticks]
        vmin, vmax = cbar_ticks[0], cbar_ticks[-1]

    # ----- Write

    d1, d2= img.shape
    im = ax.imshow(img,
                   interpolation=interpolation,
                   aspect=plot_aspect,
                   vmin=vmin,
                   vmax=vmax,
                   cmap=plot_colormap,
                   extent=[0,d2,0,d1])
    if cbar_enable is False:
        return im

    divider = make_axes_locatable(ax)

    if cbar_direction is None:
        if cbar_orientation == 'horizontal':
            cbar_direction ='bottom'
        elif cbar_orientation == 'vertical':
            cbar_direction ='right'
    cax = divider.append_axes(cbar_direction,
                          size=cbar_size,
                          pad=cbar_pad)

    colorbar(im,
            cax=cax,
            cbar_orientation=cbar_orientation,
            cbar_spacing=cbar_spacing,
            cbar_ticks=None,
            cbar_ticks_number=5)


    if fig_pdf_var is not None:
        fig_pdf_var.savefig()
        plt.close()
    return


def comparison_metric(array,
                        option='corr',
                        cbar_share=False,
                        plot_normalize=False,
                        remove_small_val=False,
                        remove_small_val_th=3
                        ):
        if option =='corr':
            title_prefix = 'Local correlation: '
            if remove_small_val:
                Cn, _ = correlation_pnr(array,
                                    remove_small_val=remove_small_val,
                                    remove_small_val_th=remove_small_val_th)
            else:
                Cn = local_correlations_fft(array)

        elif option =='var':
            #if not cbar_share:
            if plot_normalize:
                Cn = array.var(2)/array.shape[2]
            else:
                Cn = array
            title_prefix = 'Pixel variance: '

        elif option =='pnr':
            _, Cn =correlation_pnr(array,
                                    remove_small_val=remove_small_val,
                                    remove_small_val_th=remove_small_val_th)
            title_prefix = 'PNR: '

        elif option=='input':
            if plot_normalize: # not cbar_share:
                Cn =array - array.min()
                Cn = Cn/Cn.max()
            else:
                Cn = array
            title_prefix = 'Frame: '

        elif option=='snr':
            Cn1 = array.std(2)
            Cn2 = utils.noise_level(array)
            Cn = Cn1/Cn2
            title_prefix = 'SNR: '
        else:
            Cn = np.zeros(array[:2].shape())
            title_prefix = ''

        print ('%s range [%.1e %.1e]'%(title_prefix,
                                   Cn.min(),
                                   Cn.max()))
        return Cn, title_prefix


def comparison_plot(cn_see,
                    axarr=None,
                    option='corr',
                    plot_normalize=True,
                    plot_aspect=None,
                    plot_add_residual=True,
                    plot_show=True,
                    plot_orientation='horizontal',
                    plot_colormap='jet',
                    plot_num_samples=1000,
                    cbar_orientation='vertical',
                    cbar_indiv_range=None,
                    cbar_ticks_number=5,
                    cbar_spacing='uniform',
                    cbar_enable=True,
                    cbar_share=False,
                    title=True,
                    title_suffix='',
                    title_prefix=None,
                    titles_='',
                    fig_pdf_var=None,
                    remove_small_val_th=3,
                    remove_small_val=False,
                    plot_size = 10,
                    save_fig=False,
                    save_fig_name='corr_'):
    """
    """
    num_plots = len(cn_see)

    if num_plots == 2 and plot_add_residual:
        cn_see.append(cn_see[0] - cn_see[1])
        num_plots += 1

    Cn_all = []
    
    if cbar_share:
        plot_normalize=True
    for cn in cn_see:
        Cn, title_prefix2 = comparison_metric(cn,
                                            option=option,
                                            remove_small_val=remove_small_val,
                                            remove_small_val_th=remove_small_val_th,
                                            cbar_share=cbar_share,
                                            plot_normalize=plot_normalize
                                            )
        Cn_all.append(Cn)

    if title_prefix is None:
        title_prefix = title_prefix2
    #########

    if titles_=='':
        if num_plots==2:
            titles_ = ['Raw ','Denoised ']
        elif num_plots == 3:
            titles_ = ['Raw ','Denoised ','Residual']
    else:
        if len(titles_) < num_plots:
            titles_.append('Residual')

    #-----------------------------
    # Plot characteristics

    if plot_orientation == 'horizontal':
        d1, d2 = num_plots, 1
        sharex = True
        sharey = False

    elif plot_orientation =='vertical':
        d1, d2 = 1, num_plots
        sharex = False
        sharey = True

    # ---- Plot configuration
    vmax_ = list(map(np.max,Cn_all))
    vmin_ = list(map(np.min,Cn_all))

    if cbar_share:
        vmax_ = [max(vmax_)]*len(Cn_all)
        vmin_ = [min(vmin_)]*len(Cn_all)

    if cbar_indiv_range is not None:
        for ii, range_ in enumerate(cbar_indiv_range):
            vmin_[ii] = range_[0]
            vmax_[ii] = range_[1]

    dim2, dim1 = Cn.shape
    x_ticks = np.linspace(0,dim1,5).astype('int')
    y_ticks = np.linspace(0,dim2,5).astype('int')

    if axarr is None:
        fig, axarr = plt.subplots(d1,d2,
                                  figsize=(d1*plot_size,
                                           d2*plot_size),
                                  sharex=sharex,
                                  sharey=sharey)

    cbar_enable_indiv = cbar_enable and not cbar_share

    for ii, Cn in enumerate(Cn_all):
        im = show_img(Cn,
                 ax =axarr[ii],
                 cbar_orientation=cbar_orientation,
                 vmin=vmin_[ii],
                 vmax=vmax_[ii],
                 plot_aspect=plot_aspect,
                 plot_colormap=plot_colormap,
                 cbar_ticks_number=cbar_ticks_number,
                 cbar_enable=cbar_enable_indiv)

        axarr[ii].set_xticks(x_ticks)
        axarr[ii].set_yticks(y_ticks)
        axarr[ii].set_xticklabels([])
        axarr[ii].set_yticklabels([])

        if title:
            axarr[ii].set_title(title_prefix
                                + titles_[ii]
                                + title_suffix)

    if cbar_share:
        if cbar_enable:
            if len(axarr)>len(cn_see):
                cax = axarr[-1]
            else:
                cax=None
            colorbar(im,
                    cax=cax,
                    cbar_orientation=cbar_orientation
                    )
        else:
            return im
    else:
        plt.tight_layout()

    if save_fig:
        if fig_pdf_var is None:
            save_fig_name = save_fig_name+'comparison_plot_'+'.pdf'
            plt.savefig(save_fig_name)
        else:
            fig_pdf_var.savefig()
        plt.close()
    else:
        if plot_show:
            plt.show()
    return


def extract_superpixels(Yd,
                        cut_off_point=0.7,
                        length_cut=5,
                        th=2,
                        bg=False,
                        patch_size=[100,100],
                        residual_cut = 0.2,
                        low_rank=False,
                        hals=False):
    """
    """
    if Yd.min() <0:
        Yd -= Yd.min()
    if th >0:
        Yt = sup.threshold_data(Yd, th=th);
    else:
        Yt = Yd

    dims = Yt.shape[:2];
    T = Yt.shape[2];
    num_plane = 1
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    height_num = int(np.ceil(dims[0]/patch_height))
    width_num = int(np.ceil(dims[1]/(patch_width*num_plane)))
    num_patch = height_num*width_num
    patch_ref_mat = np.array(range(num_patch)).reshape(height_num,width_num, order="F")

    # find superpixel
    connect_mat_1, \
    idx, \
    comps, \
    permute_col = sup.find_superpixel(Yt,
                                      cut_off_point,
                                      length_cut,
                                      eight_neighbours=True)
    # rank1-svd
    c_ini, a_ini, ff, fb = sup.spatial_temporal_ini(Yt,
                                                    comps,idx,
                                                    length_cut,bg=bg)

    # unique superpixels
    unique_pix = np.asarray(np.sort(np.unique(connect_mat_1))[1:])
    pure_pix = []

    for kk in range(num_patch):
        pos = np.where(patch_ref_mat==kk)
        up=pos[0][0]*patch_height
        down=min(up+patch_height, dims[0])
        left=pos[1][0]*patch_width
        right=min(left+patch_width, dims[1])
        unique_pix_temp, M = sup.search_superpixel_in_range(
            (connect_mat_1.reshape(dims[0],
                                   int(dims[1]/num_plane),
                                   num_plane,order="F"))
                                    [up:down,left:right],
            permute_col, c_ini)
        pure_pix_temp = sup.fast_sep_nmf(M, M.shape[1], residual_cut)
        if len(pure_pix_temp)>0:
            pure_pix = np.hstack((pure_pix,
                                  unique_pix_temp[pure_pix_temp]))

    pure_pix = np.unique(pure_pix)

    return connect_mat_1, unique_pix, pure_pix


def superpixel_plotpixel(connect_mat_1,
                         unique_pix,
                        pure_pix,
                        ax1=None,
                        plot_aspect=None,
                        text=False,
                        plot_colormap='jet',
                        type='pure'):
    """
    """
    if ax1 is None:
        scale = np.maximum(1, (connect_mat_1.shape[1]/connect_mat_1.shape[0]));
        fig = plt.figure(figsize=(16*scale,8));
        ax1 = fig.add_subplot(111)


    if type is 'pure':
        dims = connect_mat_1.shape;
        connect_mat_1 = connect_mat_1.reshape(np.prod(dims), order="F")
        connect_mat_1[~np.in1d(connect_mat_1, pure_pix)]=0
        connect_mat_1 = connect_mat_1.reshape(dims, order="F")

    ax1.imshow(connect_mat_1,
        cmap=plot_colormap,
        aspect=plot_aspect)

    if text:
        for ii in range(len(pure_pix)):
            pos = np.where(connect_mat_1[:,:] == pure_pix[ii])
            pos0 = pos[0];
            pos1 = pos[1];
            ax1.text((pos1)[np.array(len(pos1)/3,dtype=int)],
                    (pos0)[np.array(len(pos0)/3,dtype=int)],
                    f"{np.where(unique_pix==pure_pix[ii])[0][0]}",
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    color='white',
                    fontsize=15,
                    fontweight="bold")
    return


def superpixel_component(Yd,
                        cut_off_point=0.9,
                        length_cut=10,
                        th=2,
                        num_plane=1,
                        plot_en=False,
                        text=False):

    dims = Yd.shape[:2]
    T = Yd.shape[2]

    Yt = sup.threshold_data(Yd, th=th)
    connect_mat_1, idx, comps, permute_col = sup.find_superpixel(Yt,
                                                                cut_off_point,
                                                                length_cut,
                                                                eight_neighbours=True)

    return Yt, connect_mat_1, idx, comps, permute_col


def correlation_pnr(Y,
                    remove_small_val =False,
                    remove_small_val_th =3
                   ):
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image.
    """
    # compute peak-to-noise ratio
    Y = Y - Y.mean(2, keepdims=True)
    data_max = Y.max(2)
    data_std = utils.noise_level(Y)
    # compute PNR image
    pnr = np.divide(data_max, data_std)

    if remove_small_val:
        pnr[np.abs(pnr) < 0] = 0

    tmp_data = Y / data_std[:,:,np.newaxis]

    if remove_small_val:
        tmp_data[np.abs(tmp_data) < remove_small_val_th] = 0

    # compute correlation image
    cn = local_correlations_fft(tmp_data, swap_dim=True)

    return cn, pnr


def local_correlations_fft(Y,
                            eight_neighbours=True,
                            swap_dim=True,
                            opencv=True):
    """Computes the correlation image for the input
    dataset Y using a faster FFT based method

    Parameters:
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format

    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively

    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    opencv: Boolean
        If True process using open cv method

    Returns:
    --------
    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """

    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = Y.copy()
        for idx, img in enumerate(Yconv):
            Yconv[idx] = cv2.filter2D(img, -1, sz, borderType=0)
        MASK = cv2.filter2D(
            np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = filters.convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = filters.convolve(
            np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    Cn = np.mean(Yconv * Y, axis=0) / MASK
    return Cn



def cn_ranks_sum_plot(ranks,
                    dims,
                    nblocks=[10, 10],
                    ax=None,
                    plot_aspect=None,
                    figsize=5,
                    fig_pdf_var=None,
                    list_order='C',
                    plot_colormap='YlGnBu',
                    cbar_orientation='vertical'
                    ):
    """
    """

    Cplots, title_  = cn_ranks_sum(ranks,
                                    dims,
                                    nblocks=nblocks,
                                    list_order=list_order
                                    )
    # Call plotting function
    title_ = 'Compression ratio %.2f'%(num_pixels/rank_sum)
    show_img(Cplots,
             ax=ax,
             cbar_orientation=cbar_orientation,
             plot_aspect=plot_aspect,
             plot_colormap=plot_colormap,
             plot_size=(d1 * figsize, d2 * figsize),
             cbar_ticks_number=5,
             cbar_ticks=None,
             title_=title_,
             fig_pdf_var=fig_pdf_var,
             cbar_enable=True)
    return


def cn_ranks_dx_plot(ranks,
                    dims,
                    nblocks=[10, 10],
                    figsize=3,
                    fontsize=10,
                    tile_err=100,
                    text_en=True,
                    include_err=True,
                    fig_pdf_var=None,
                    fig_cmap = 'YlGnBu',
                    list_order='C',
                    plot_aspect='auto',
                    save_fig=False,
                    save_fig_name=''):
    """
    2X2 grid for 4dx denoiser
    """
    rtype=[None, 'r', 'c', 'rc']

    d1, d2 = dims[:2] // np.min(dims[:2])
    d1, d2 = min(d1,5), min(d2,5)

    assert(len(ranks)==4)
    widths = [1]
    heights =[0.25]*4
    a, b = 4, 1
    gs_kw = dict(width_ratios=widths, height_ratios=heights,hspace=0.7)
    fig, axall = plt.subplots(a, b,
                            figsize=(figsize*d2*b,a*figsize*d1),
                            gridspec_kw=gs_kw)

    for ii, rank in enumerate(ranks):
        cname_ = save_fig_name+'_offset_'+str(rtype[ii])+'_'
        # update according to master
        if not include_err:
            rank = rank % tile_err
            rank[rank==0] = 1
        _ = cn_ranks_plot(np.asarray(rank),
                        dims,
                        nblocks=nblocks,
                        ax3=axall.flatten(order='F')[ii],
                        offset_case=rtype[ii],
                        figsize=figsize,
                        fig_cmap=fig_cmap,
                        fontsize=fontsize,
                        text_en=text_en,
                        plot_aspect=plot_aspect,
                        list_order=list_order,
                        save_fig=save_fig,
                        save_fig_name=cname_)

    if fig_pdf_var is None:
        plt.show()
    else:
        fig_pdf_var.savefig()
    return


def plot_comp(Y, Y_hat=None, title_=None, dims=None, idx_=0,dim_order='F'):
    """
    Plot comparison for frame idx_ in Y, Y_hat.
    Y dxT to be reshaped to dims=(d1,d2,T)
    """
    if Y_hat is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        if title_ is not None:
            ax.set_title(title_)
        plots_ = zip([ax], [Y])
    else:
        R = Y - Y_hat
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        if title_ is not None:
            print(title_)
            ax[0].set_title(title_)
        plots_ = zip(ax, [Y, Y_hat, R])

    for ax_, arr in plots_:
        if np.ndim(arr) > 2:
            arr = arr.reshape(dims, order=dim_order)[:, :, idx_]
        else:
            arr = arr.reshape(dims[:2], order=dim_order)
        show_img(arr,
                 ax =ax_,
                 plot_colormap='viridis',
                cbar_ticks_number=4)
        ax_.set_title(title_)

    plt.show()
    return


def plot_temporal_traces(V_TF, V_hat=None,title_=''):
    """
    """
    if np.ndim (V_TF)==1:
        V_TF =V_TF[np.newaxis,:]
    if np.ndim (V_hat)==1:
        v_hat = V_hat[np.newaxis,:]
    else:
        v_hat = V_hat

    for idx, vt in enumerate(np.asarray(V_TF)):
        plt.figure(figsize=(15, 5))
        plt.title(('Temporal component %d'+ title_)% idx)
        plt.plot(vt, 'b-')
        if v_hat is not None:
            plt.plot(v_hat[idx, :], 'r--')
            plt.legend(['raw','denoised'])
        plt.show()
    return


def plot_spatial_component(U_,
                           Y_hat=None,
                           dims=None,
                           dim_order='F'):
    """
    """
    if np.ndim(U_) ==1:
        U_=U_[:,np.newaxis]

    if np.ndim(Y_hat) ==1:
        Y_hat=Y_hat[:,np.newaxis]

    U_hat_c = None
    n_components = U_.shape[1]
    for ii in range(n_components):
        if Y_hat is not None:
            U_hat_c = Y_hat[:,ii]
        plot_comp(U_[:, ii],
            Y_hat=U_hat_c,
            dim_order=dim_order,
            title_='Spatial component U' +str(ii),
                  dims=dims[:2])
    return


def nearest_frame_corr(A):
    """
    """
    num_frames = A.shape[2]
    corrs = np.zeros((num_frames-1,))
    for idx in range(num_frames-1):
        frame1 = A[:,:,idx].flatten()
        frame2 = A[:,:,idx+1].flatten()
        corrs[idx] =  corr(frame1,frame2)

    return corrs


def corr(a,b):
    a -= a.mean()
    b -= b.mean()
    return a.dot(b) / sqrt(a.dot(a) * b.dot(b) + np.finfo(float).eps)


def correlation_traces(Y,Yd,
                    fig_size=(10,6),
                    fig_pdf_var=None):
    """
    compute correlations between nearest
    neighbor frames and show these as a trace.
    """
    R = Y - Yd
    corrs_Y = nearest_frame_corr(Y)
    corrs_Yd = nearest_frame_corr(Yd)
    corrs_R = nearest_frame_corr(R)

    # plot Y and Yd on the same scale
    max_scale = max(corrs_Y.max(),corrs_Yd.max(),corrs_R.max())
    min_scale = min(corrs_Y.min(),corrs_Yd.min(),corrs_R.min())

    # assign these to be in the same scale
    fig = plt.figure(figsize=fig_size)
    plt.title('Correlation Traces')
    plt.plot(corrs_Y)
    plt.plot(corrs_Yd)
    plt.plot(corrs_R)
    plt.ylim(min_scale, max_scale)
    plt.legend(['raw','denoised','residual'])

    if fig_pdf_var is None:
        plt.show()
    else:
        fig_pdf_var.savefig(fig)
        plt.close()
    return


def snr_per_frame(Y,Yd,R,
                    cbar_orientation='vertical',
                    plot_orientation='horizontal',
                    title=True,
                    titles_=['SNR_frame']):
    """
    take a patch and sum Y, Yd, and R over all pixels to get three traces.
    """
    titles_ = titles_*3
    titles_[0].append(' raw')
    titles_[1].append(' denoised')
    titles_[2].append(' residual')

    Ys = Y.sum(2)
    Yds = Yd.sum(2)
    Rs = R.sum(2)
    comparison_plot([Ys,Yds,Rs],
                    cbar_orientation=cbar_orientation,
                    option='input',
                    plot_orientation=plot_orientation,
                    title=title,
                    titles_=titles_)
    return


def nearest_frame_corr(A):
    """
    """
    num_frames = A.shape[2]
    corrs = np.zeros((num_frames-1,))
    for idx in range(num_frames-1):
        frame1 = A[:,:,idx].flatten()
        frame2 = A[:,:,idx+1].flatten()
        corrs[idx] =  corr(frame1,frame2)
    return corrs


def corr(a,b):
    a -= a.mean()
    b -= b.mean()
    return a.dot(b) / np.sqrt(a.dot(a) * b.dot(b) + np.finfo(float).eps)


def spatial_filter_spixel_plot(data,y_hat,hat_k):
    """
    """
    Cn_y, _ = correlation_pnr(data)
    Cn_yh,_ = correlation_pnr(y_hat)

    fig, ax = plt.subplots(1,3,figsize=(10,5))
    im0 = ax[0].imshow(Cn_y.T,vmin=maps[0],vmax=maps[1])
    if neuron_indx is None:
        im1 = ax[1].imshow(hat_k)
    else:
        im1 = ax[1].imshow(hat_k[:,np.newaxis].T)
    im2 = ax[2].imshow(Cn_yh.T,vmin=maps[0],vmax=maps[1])
    ax[0].set_title('y')
    ax[1].set_title('k')
    ax[2].set_title('y_hat')

    ax[0].set_xticks(np.arange(y_hat.shape[0]))
    ax[0].set_yticks(np.arange(y_hat.shape[1]))
    ax[2].set_xticks(np.arange(y_hat.shape[0]))
    ax[2].set_yticks(np.arange(y_hat.shape[1]))
    ax[1].set_yticks(np.arange(1))

    if neuron_indx is None:
        ax[1].set_xticks(np.arange(np.prod(y_hat.shape[:2]))[::4])
        ax[1].set_yticks(np.arange(np.prod(y_hat.shape[:2]))[::4])

    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("bottom", size="5%", pad=0.5)
    cbar0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.5)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f", orientation='horizontal')
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("bottom", size="5%", pad=0.5)
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
    plt.tight_layout()
    plt.show()
    return


def intialization_plot(data_highpass,
                       patch_radius=20,
                       min_pnr=0,
                       min_corr=0,
                       stdv_pixel=None,
                       noise_thresh=3,
                       orientation='horizontal'):
    """
    """

    if orientation == 'horizontal':
        d1, d2 = 2, 1
    elif orientation == 'vertical':
        d1, d2 = 1, 2
    fig, axarr = plt.subplots(d1, d2, figsize=(14, 7), sharex=True)

    # Compute pixel-wise noise stdv
    if not stdv_pixel:
        stdv_pixel = np.sqrt(np.var(data_highpass, axis=-1))

    # Compute & plot corr image
    data_spikes = data_highpass - \
        np.median(data_highpass, axis=-1)[:, :, np.newaxis]
    data_spikes[data_spikes < noise_thresh * stdv_pixel[:, :, np.newaxis]] = 0
    corr_image = local_correlations_fft(
        data_spikes.transpose([2, 0, 1]), swap_dim=False)

    if min_corr:
        corr_image[corr_image < min_corr] = 0
    show_img(axarr[0], corr_image, orientation=orientation)
    axarr[0].set_title('Thresholded Corr Image')

    # Compute & plot pnr image
    pnr_image = np.divide(np.max(data_highpass, axis=-1),
                          stdv_pixel)
    pnr_image[np.logical_or(corr_image < min_corr, pnr_image < min_pnr)] = 0
    pnr_image = filters.median_filter(pnr_image,
                                      size=(int(round(patch_radius / 4)),) * 2,
                                      mode='constant')
    show_img(axarr[1], pnr_image, orientation=orientation)
    axarr[1].set_title('Thresholded & Filtered PNR Image')

    # Display PLot
    plt.tight_layout()
    plt.show()
    return
