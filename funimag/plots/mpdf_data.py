import numpy as np
import matplotlib.patches as patches
from . import util_plot
from .. import denoise


def trace_characteristics(ax,
                          legend=None,
                          yticks=True):
    """
    """
    if legend is not None:
        ax.legend(legend,
                  borderaxespad=0,
                  handlelength=1.0,
                    loc='upper right',
                    bbox_to_anchor=(1.15,1.15),
                    fancybox=True,
                    shadow=True)
    if yticks:
        ax.margins(0)
        yticks = ax.get_ylim()
        midpoint = (max(yticks)+min(yticks))/2
        yticks =[(midpoint+min(yticks))/2,
                 (midpoint+max(yticks))/2,
                ]
        yticks = [np.round(x,2) for x in yticks]
        yticklabels = ['{:.2f}'.format(x) for x in yticks]
        ax.set_yticks(yticks)
        ax.set_xmargin(0.01)
        
    return


def trace_extract(Y,Yd,R,a,b,trace_seg=[0,1000]):
    """
    """
    trace_seg_a, trace_seg_b = trace_seg
    t1 = Y[a,b,trace_seg_a:trace_seg_b]
    t2 = Yd[a,b,trace_seg_a:trace_seg_b]
    t3 = R[a,b,trace_seg_a:trace_seg_b]
    offset = min(t2.min(),t1.min())
    scale = max((t2 -offset).max(), (t1 -offset).max())
    trace1 = (t1 - offset)/scale
    trace2 = (t2 - offset)/scale
    trace3 = trace1-trace2
    return trace1, trace2, trace3


def extract_frame (x,al,au,bl,bu,frame_idx):
    """
    """
    a = x[al:au,bl:bu,frame_idx]
    return a


def box_lim(a1, b1, dims, zoom_box=15):
    """
    """
    al1 = max(0,a1-zoom_box)
    au1 = min(dims[0],a1+zoom_box)

    bl1 = max(0 , b1-zoom_box)
    bu1 = min(dims[1],b1+zoom_box)
    return al1, au1, bl1, bu1


def plot_datain(ax,page_count,
                cplot_row,
                Y,Yd,
                nblocks=None,
                ranks=None,
                frame_idx=900,
                pixel_coor1=[10,10],pixel_coor2=[11,11],
                trace_seg=[0,1000],
                zoom_box1=20,zoom_box2=20,
                plot_colormap='jet',
                list_order='C'):
    """
    """
    #---- Plot variables
    cbar_ticks_number = 5
    trace_offset = 0.02
    
    #---- Superpixel parameters
    sup_cut_off_point1 = 0.2
    sup_cut_off_point2 = 0.7
    sup_length_cut = 10
    sup_min_threshold = 2
    sup_residual_cut = 0.2
    sup_background = False
    sup_lowrank = False
    sup_hals = False
    sup_text = False
    # ----------------------
    dims = Y.shape
    R = Y-Yd
    a1, b1 = pixel_coor1
    a2, b2 = pixel_coor2

    trace1, trace2,trace3 = trace_extract(Y,Yd,R,a1,b1,trace_seg=trace_seg)
    trace4, trace5,trace6 = trace_extract(Y,Yd,R,a2,b2,trace_seg=trace_seg)

    al, au, bl, bu = box_lim(a1, b1, dims, zoom_box=zoom_box2)
    al1, au1, bl1, bu1 = box_lim(a1, b1, dims, zoom_box=zoom_box1)
    #al2, au2, bl2, bu2 = box_lim(a2, b2, dims, zoom_box=zoom_box)

    g1trace_ub = max(trace1.max(), trace2.max(), trace3.max())
    g1trace_lb = min(trace1.min(), trace2.min(), trace3.min())
    g2trace_ub = max(trace4.max(), trace5.max(), trace6.max())
    g2trace_lb = min(trace4.min(), trace5.min(), trace6.min())

    n_ticks = 3
    y_ticks1 = np.linspace(g1trace_lb,g1trace_ub,n_ticks)
    y_ticks2 = np.linspace(g2trace_lb,g2trace_ub,n_ticks)

    trace_ub = max(g1trace_ub, g2trace_ub)
    trace_lb = min(g1trace_lb, g2trace_lb)

    trace_color_raw ='dimgray'
    trace_color_denoised ='navy'
    trace_color_residual = 'darkslategrey'
    trace_line_style = '-'

    #################### PAGE 1
    if page_count == 1:

        if cplot_row==1:
            t1,t2 = trace_seg
            C1 = Y[al1:au1,bl1:bu1,t1:t2]
            C2 = Yd[al1:au1,bl1:bu1,t1:t2]
            C3 = R[al1:au1,bl1:bu1,t1:t2]
            
            offset = np.minimum(C1.min(2),C2.min(2))
            scale = np.maximum(C1.max(2)-offset,C2.max(2)-offset)
            C1d = (C1 - offset[:,:,np.newaxis])/scale[:,:,np.newaxis]
            C2d = (C2 - offset[:,:,np.newaxis])/scale[:,:,np.newaxis]
            C3d = C1d - C2d
            
            C1d =C1d[:,:,frame_idx-t1]
            C2d =C2d[:,:,frame_idx-t1]
            C3d =C3d[:,:,frame_idx-t1]
            cin = [C1d,C2d,C3d]
            
            d1,d2= cin[0].shape
            frame_str=str(frame_idx-trace_seg[0])+' '
            mov_type =['Raw','Denoised','Residual']
            titles_=[frame_str + each_title for each_title in mov_type]
            util_plot.comparison_plot(cin,
                                      plot_show=False,
                                      option='input',
                                      axarr=ax,
                                      titles_=titles_,
                                      cbar_ticks_number=cbar_ticks_number,
                                      cbar_enable=True,
                                      cbar_share=True,
                                      plot_aspect='auto')

            for myax in ax[:3]:
                myax.set_yticks([])
                myax.scatter(b1-bl1,
                            au1-a1,
                            edgecolors='k',
                            marker='o',
                            lw=2,
                            facecolors='None')
                

                myax.scatter(b2-bl1,
                            au1-a2,
                            edgecolors='dimgray',
                            marker='o',
                            lw=2,
                            facecolors='None')


            for myax in ax[:3]:
                myax.set_xticks([])

        #---- Traces
        elif cplot_row==2:
            ax.plot(np.arange(len(trace1)),
                    trace1, c=trace_color_raw,ls=trace_line_style)
            ax.plot(np.arange(len(trace2)),
                    trace2, c=trace_color_denoised,ls=trace_line_style)
            legend_ = ['raw','denoised']
            trace_characteristics(ax,legend=legend_)
            ax.set_xticks([])

        elif cplot_row ==3:
            ax.plot(np.arange(len(trace3)),trace3,
                    c=trace_color_residual,
                    ls=trace_line_style)
            legend_= ['residual']
            trace_characteristics(ax,legend=legend_)
            ax.set_xticks([])
            
        elif cplot_row ==4:
            ax.plot(np.arange(len(trace4)),trace4,
                    c=trace_color_raw,
                    ls=trace_line_style)
            ax.plot(np.arange(len(trace5)),trace5,
                    c=trace_color_denoised,
                    ls=trace_line_style)
            ax.set_xticks([])
            trace_characteristics(ax)

        elif cplot_row ==5:
            ax.plot(trace6, c=trace_color_residual,ls=trace_line_style)
            trace_characteristics(ax)
            ax.set_xlabel('Frames')
        return

    #---- PAGE 2

    if page_count == 2:
        if cplot_row == 1:
            t1,t2 = trace_seg
            C1 = Y[al:au,bl:bu,:]
            C2 = Yd[al:au,bl:bu,:]
            C3 = R[al:au,bl:bu,:]
            
            offset = np.minimum(C1.min(2),C2.min(2))
            scale = np.maximum(C1.max(2)-offset,C2.max(2)-offset)
            A = (C1 - offset[:,:,np.newaxis])/scale[:,:,np.newaxis]
            B = (C2 - offset[:,:,np.newaxis])/scale[:,:,np.newaxis]
            C = A - B
            
            cin =[A,B,C]
            util_plot.comparison_plot(cin,
                                      plot_show=False,
                                      option='snr',
                                      axarr=ax,
                                      cbar_enable=True,
                                      cbar_share=True,
                                      plot_add_residual=False,
                                      plot_colormap=plot_colormap,
                                      plot_aspect='auto')
            for myax in ax[:4]:
                myax.set_xticks([])
                myax.set_yticks([])
            
            if not zoom_box1==zoom_box2:
                cy = al1-al
                cx = bl1-bl
                cyy = au1-al1 
                cxx = bu1-bl1
                for myax in ax[:3]:
                    rect = patches.Rectangle((cx,cy),
                                             cxx,cyy,
                                             linewidth=2,
                                             edgecolor='r',
                                             facecolor='none')
                    myax.add_patch(rect)
        if cplot_row == 2:
            
            t1,t2 = trace_seg
            C1 = Y[al:au,bl:bu,:]
            C2 = Yd[al:au,bl:bu,:]
            
            offset = np.minimum(C1.min(2),C2.min(2))
            scale = np.maximum(C1.max(2)-offset,C2.max(2)-offset)
            A = (C1 - offset[:,:,np.newaxis])/scale[:,:,np.newaxis]
            B = (C2 - offset[:,:,np.newaxis])/scale[:,:,np.newaxis]
            
            A1 = util_plot.comparison_metric(A, option='snr')[0]
            B1 = util_plot.comparison_metric(B, option='snr')[0]
            D1 = B1/A1
            cin =[D1]
            util_plot.show_img(D1,
                             ax=ax,
                             cbar_orientation='vertical',
                             cbar_direction='left',
                             cbar_size="2%",
                             cbar_pad=0.1,
                             plot_aspect='auto',
                             plot_colormap=plot_colormap,
                             cbar_ticks_number=cbar_ticks_number,
                             cbar_enable=True)
            ax.set_title('SNR Ratio')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if not zoom_box1==zoom_box2:
                cy = al1-al
                cx = bl1-bl
                cyy = au1-al1 
                cxx = bu1-bl1
                rect = patches.Rectangle((cx,cy),
                                         cxx,cyy,
                                         linewidth=2,
                                         edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)

        elif cplot_row == 3:
            Y1=Y[al:au,bl:bu]
            R1=R[al:au,bl:bu]

            util_plot.comparison_plot([Y1,R1],
                                plot_show=False,
                                plot_aspect='auto',
                                plot_add_residual=False,
                                plot_colormap=plot_colormap,
                                cbar_orientation='vertical',
                                cbar_ticks_number=cbar_ticks_number,
                                cbar_share=True,
                                titles_=['Raw','Residual'],
                                axarr=ax)

            for myax in ax[:2]:
                myax.set_xticks([])
                myax.set_yticks([])

            if not zoom_box1==zoom_box2:
                cy = al1-al
                cx = bl1-bl
                cyy = au1-al1 
                cxx = bu1-bl1
                for myax in ax[:2]:
                    rect = patches.Rectangle((cx,cy),
                                             cxx,cyy,
                                             linewidth=2,
                                             edgecolor='r',
                                             facecolor='none')
                    myax.add_patch(rect)
             

        elif cplot_row ==4:
            Y1=Y[al:au,bl:bu,:]
            connect_mat_1, \
            unique_pix, \
            pure_pix = util_plot.extract_superpixels(Y1,
                                                    cut_off_point=sup_cut_off_point1,
                                                    length_cut=sup_length_cut,
                                                    th=sup_min_threshold,
                                                    bg=sup_background,
                                                    residual_cut =sup_residual_cut,
                                                    low_rank=sup_lowrank,
                                                    hals=sup_hals)

            util_plot.superpixel_plotpixel(connect_mat_1,
                                            unique_pix,
                                            pure_pix,
                                            plot_aspect='auto',
                                            plot_colormap=plot_colormap,
                                            ax1=ax,
                                            text=sup_text)

            ax.set_title('Raw\n(Cut %.1f,Len %d)'%(sup_cut_off_point1,
                                                sup_length_cut))

        elif cplot_row ==5:
            Yd1=Yd[al:au,bl:bu,:]
            connect_mat_1, unique_pix, pure_pix = util_plot.extract_superpixels(Yd1,
                                                    cut_off_point=sup_cut_off_point2,
                                                    length_cut=sup_length_cut,
                                                    th=sup_min_threshold,
                                                    bg=sup_background,
                                                    residual_cut =sup_residual_cut,
                                                    low_rank=sup_lowrank,
                                                    hals=sup_hals)

            util_plot.superpixel_plotpixel(connect_mat_1,
                                            unique_pix,
                                            pure_pix,
                                            ax1=ax,
                                            plot_aspect='auto',
                                            plot_colormap=plot_colormap,
                                            text=sup_text)
            ax.set_title('Denoised\n(Cut %.1f,Len %d)'%(sup_cut_off_point2,
                                                sup_length_cut))
        elif cplot_row == 10:
            _=util_plot.cn_ranks_plot(ranks,
                                    dims=dims,
                                    cratio_tile=True,
                                    cbar_pad=0.1,
                                    cbar_size="3%",
                                    nblocks=nblocks,
                                    ax3=ax,
                                    grid_cut=[al,au,bl,bu],
                                    cbar_orientation='vertical',
                                    cbar_direction='left',
                                    plot_aspect='auto',
                                    fig_cmap='YlGnBu',
                                    text_en=False)
            ax.set_xticks([])
            ax.set_yticks([])

        else:
            pass 
        return