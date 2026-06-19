import cortex, os, shutil, warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def _pycortex_svg_overlays_available():
    return shutil.which("inkscape") is not None


def _load_fsaverage_roi_overlay(roi_overlay, nsd_dir=None, roi_defs_dir=None):
    if roi_overlay is None:
        return None, None
    if isinstance(roi_overlay, str):
        if roi_defs_dir is None:
            if nsd_dir is None:
                raise ValueError(
                    "Pass nsd_dir or roi_defs_dir when roi_overlay is a named ROI file "
                    "(for example roi_overlay='streams')."
                )
            roi_defs_dir = os.path.join(nsd_dir, "nsddata/freesurfer/fsaverage/label")
        from nsd_visuo_semantics.utils.nsd_get_data_light import get_rois
        return get_rois(roi_overlay, roi_defs_dir)

    return np.asarray(roi_overlay), None


def _add_fsaverage_roi_contours(fig, roi_data, height=480, roi_ids=None,
                                linecolor="black", linewidth=0.8, alpha=0.95):
    if roi_data is None:
        return

    roi_data = np.asarray(roi_data).squeeze()
    if roi_data.shape[0] != 327684:
        raise ValueError(
            f"Expected an fsaverage ROI vector with 327684 vertices, got shape {roi_data.shape}."
        )

    labels = np.unique(roi_data[np.isfinite(roi_data)])
    labels = labels[labels != 0]
    if roi_ids is not None:
        labels = np.asarray([label for label in labels if label in roi_ids])

    if labels.size == 0:
        warnings.warn("No non-zero ROI labels found to contour.", RuntimeWarning)
        return

    from cortex.quickflat.utils import make_flatmap_image

    ax = fig.axes[0]
    for label in labels:
        roi_mask = (roi_data == label).astype(np.float32)
        mask_vertex = cortex.dataset.Vertex(roi_mask, "fsaverage", cmap="gray", vmin=0, vmax=1)
        mask_img, extents = make_flatmap_image(mask_vertex, height=height)
        mask_img = np.nan_to_num(mask_img, nan=0.0)
        if np.nanmax(mask_img) < 0.5:
            continue
        ax.contour(
            mask_img,
            levels=[0.5],
            colors=[linecolor],
            linewidths=linewidth,
            alpha=alpha,
            origin="upper",
            extent=extents,
            zorder=1100,
        )


def pyplot_brain(fsavg_data, savename, figpath, save_type='png', max_cmap_val=None,
                 with_rois=False, with_sulci=False, with_labels=True, roi_list=None,
                 roi_overlay=None, nsd_dir=None, roi_defs_dir=None, roi_ids=None,
                 roi_linecolor="black", roi_linewidth=0.8):

    os.makedirs(figpath, exist_ok=True)

    if max_cmap_val is None:
        boundar = np.nanmax(np.abs(fsavg_data))
    else:
        boundar = np.nanmax(np.abs(max_cmap_val))

    vert = cortex.dataset.Vertex(fsavg_data, "fsaverage", cmap='RdBu_r', vmin=-boundar, vmax=boundar)
    requested_svg_overlay = with_rois or with_sulci
    if requested_svg_overlay and not _pycortex_svg_overlays_available():
        warnings.warn(
            "Pycortex ROI/sulci overlays require Inkscape to rasterize SVG overlays. "
            "Inkscape was not found, so plotting will continue without ROI/sulci overlays. "
            "Install inkscape and restart the kernel to enable overlays.",
            RuntimeWarning,
        )
        with_rois = False
        with_sulci = False

    try:
        flatmap = cortex.quickflat.make_figure(
            vert,
            height=480,
            with_colorbar=1,
            with_rois=with_rois,
            with_sulci=with_sulci,
            with_labels=with_labels,
            roi_list=roi_list,
            linecolor="black",
            linewidth=1,
            roifill=(0, 0, 0, 0),
        )
    except RuntimeError as exc:
        if "Inkscape" not in str(exc) or not requested_svg_overlay:
            raise
        warnings.warn(
            "Pycortex failed while rendering ROI/sulci overlays with Inkscape. "
            "Retrying without ROI/sulci overlays.",
            RuntimeWarning,
        )
        flatmap = cortex.quickflat.make_figure(
            vert,
            height=480,
            with_colorbar=1,
            with_rois=False,
            with_sulci=False,
        )

    roi_data, _ = _load_fsaverage_roi_overlay(roi_overlay, nsd_dir=nsd_dir, roi_defs_dir=roi_defs_dir)
    _add_fsaverage_roi_contours(
        fig=flatmap,
        roi_data=roi_data,
        height=480,
        roi_ids=roi_ids,
        linecolor=roi_linecolor,
        linewidth=roi_linewidth,
    )
    
    fig = plt.gcf()
    fig.suptitle(f'{savename} - max abs val: {np.nanmax(np.abs(fsavg_data)):.2f}')
    plt.savefig(f'{figpath}/{savename}.{save_type}', dpi=300)
    plt.close()


def pyplot_indiv_subjects(fsavg_data_allsub, savename, figpath, save_type='png',
                          max_cmap_val=None, with_rois=False, with_sulci=False,
                          with_labels=True, roi_list=None, roi_overlay=None,
                          nsd_dir=None, roi_defs_dir=None, roi_ids=None,
                          roi_linecolor="black", roi_linewidth=0.8):
    '''
    fsavg_data_allsub: [n_subj, n_fsavg_voxels]'''
    
    for s in range(fsavg_data_allsub.shape[0]):
        fsavg_data = fsavg_data_allsub[s]
        pyplot_brain(
            fsavg_data,
            f'{savename}_subj{s+1:02}',
            figpath,
            save_type=save_type,
            max_cmap_val=max_cmap_val,
            with_rois=with_rois,
            with_sulci=with_sulci,
            with_labels=with_labels,
            roi_list=roi_list,
            roi_overlay=roi_overlay,
            nsd_dir=nsd_dir,
            roi_defs_dir=roi_defs_dir,
            roi_ids=roi_ids,
            roi_linecolor=roi_linecolor,
            roi_linewidth=roi_linewidth,
        )


def pyplot_subj_avg(fsavg_data_allsub, savename, figpath, sig_mask='fdr_bh',
                    save_type='png', max_cmap_val=None, with_rois=False,
                    with_sulci=False, with_labels=True, roi_list=None,
                    roi_overlay=None, nsd_dir=None, roi_defs_dir=None,
                    roi_ids=None, roi_linecolor="black", roi_linewidth=0.8):
    '''
    fsavg_data_allsub: [n_subj, n_fsavg_voxels]
    sig_mask: "fdr_bh", "uncorrected" or None (no mask is applied if None)'''
    
    sig_str = '' if sig_mask is None else sig_mask
    
    # get mean
    fsavg_data = np.nanmean(fsavg_data_allsub, axis=0)
    
    # 2-tailed t-test against 0 for each element in fsavg_data
    if sig_mask != None:
        t_stat, p_values = ttest_1samp(fsavg_data_allsub, 0, axis=0, nan_policy='omit')
    
    # Apply significance mask if specified
    if sig_mask == 'fdr_bh':
        # Apply FDR correction
        _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        sig_mask = p_values_corrected < 0.05
    elif sig_mask == 'uncorrected':
        sig_mask = p_values < 0.05
    else:
        sig_mask = np.ones_like(fsavg_data, dtype=bool)  # No masking, all true
    
    # Apply significance mask to data
    fsavg_data_masked = np.where(sig_mask, fsavg_data, np.nan)  # Mask non-significant points with NaN

    pyplot_brain(
        fsavg_data_masked,
        f'{savename}_subjavg_{sig_str}',
        figpath,
        save_type=save_type,
        max_cmap_val=max_cmap_val,
        with_rois=with_rois,
        with_sulci=with_sulci,
        with_labels=with_labels,
        roi_list=roi_list,
        roi_overlay=roi_overlay,
        nsd_dir=nsd_dir,
        roi_defs_dir=roi_defs_dir,
        roi_ids=roi_ids,
        roi_linecolor=roi_linecolor,
        roi_linewidth=roi_linewidth,
    )


def get_fsavg_data_from_path(base_dir, model_name, layer_id, model_suffix='', 
                             n_subjects=8, n_vertices=327684, hemis=['lh', 'rh']):
    '''
    NOTE: we will format the datapath string using the arguments passed here. make sure
    they correspond to the way things are saved in the base_dir.
    RETURNS: an array of shape [n_subjects, n_vertices] where each row is the flattened data for a subject
    '''

    datapath = os.path.join(base_dir, '%s', model_name, '%s_correlation_fsaverage', '%s.%s-model-%s-surf.npy')

    main_data = np.zeros((n_subjects, n_vertices), dtype=np.float32)
    # loop over subjects
    for sub in range(n_subjects):
        subj = f'subj{(1+sub):02d}'
        sub_data = []
        for this_hemi in hemis:
            if 'dnn' in model_name:
                sub_data.append(np.load(datapath % (subj, model_name + model_suffix, this_hemi, subj, str(layer_id))))
            else:
                sub_data.append(np.load(datapath % (subj, model_name + model_suffix, this_hemi, subj, str(1))))
        main_data[sub, :] = np.concatenate(sub_data)

    return main_data


def pyplot_brains_from_models_list(models, contrast_models, base_dir, 
                                   layer=60, contrast_layer='same', 
                                   contrast_same_model=True, max_cmap_val=None,
                                   save_type='png', figpath='', 
                                   plot_indiv_sub=True, plot_subj_avg=True,
                                   with_rois=False, with_sulci=False,
                                   with_labels=True, roi_list=None,
                                   roi_overlay=None, nsd_dir=None,
                                   roi_defs_dir=None, roi_ids=None,
                                   roi_linecolor="black", roi_linewidth=0.8):
    '''models: list of model names to plot
    contrast_models: list of model names to use as contrast
    base_dir: base directory where the data is saved
    layer: layer to plot (only used if "dnn" in model_name)
    contrast_layer: layer to use for contrast (only used if "dnn" in model_name). can be "same" or a layer number
    contrast_same_model: if True, do not plot the same model as contrast
    max_cmap_val: maximum value for the colorbar
    save_type: file type to save the figures
    figpath: path to save the figures
    plot_indiv_sub: if True, plot individual subject maps
    plot_subj_avg: if True, plot the average map across subjects (with sig threshold)'''

    if not isinstance(models, list):
        models = [models]
    if not isinstance(contrast_models, list):
        contrast_models = [contrast_models]

    if contrast_layer == 'same':
        contrast_layer = layer
    elif isinstance(contrast_layer, int):
        pass
    else:
        raise NotImplementedError('Only "same" is implemented for contrast_layer')

    for model_name in models:

        fsavg_data = get_fsavg_data_from_path(base_dir, model_name, layer, n_subjects=1)
        if plot_indiv_sub:
            pyplot_indiv_subjects(
                fsavg_data,
                model_name,
                figpath,
                save_type=save_type,
                max_cmap_val=max_cmap_val,
                with_rois=with_rois,
                with_sulci=with_sulci,
                with_labels=with_labels,
                roi_list=roi_list,
                roi_overlay=roi_overlay,
                nsd_dir=nsd_dir,
                roi_defs_dir=roi_defs_dir,
                roi_ids=roi_ids,
                roi_linecolor=roi_linecolor,
                roi_linewidth=roi_linewidth,
            )
        if plot_subj_avg:
            pyplot_subj_avg(
                fsavg_data,
                model_name,
                figpath,
                sig_mask='fdr_bh',
                save_type=save_type,
                max_cmap_val=max_cmap_val,
                with_rois=with_rois,
                with_sulci=with_sulci,
                with_labels=with_labels,
                roi_list=roi_list,
                roi_overlay=roi_overlay,
                nsd_dir=nsd_dir,
                roi_defs_dir=roi_defs_dir,
                roi_ids=roi_ids,
                roi_linecolor=roi_linecolor,
                roi_linewidth=roi_linewidth,
            )

        for contrast_model_name in contrast_models:

            if not contrast_same_model and contrast_model_name == model_name:
                continue

            fsavg_data_contrast = get_fsavg_data_from_path(base_dir, contrast_model_name, contrast_layer)
            
            fsavg_data_diff = fsavg_data - fsavg_data_contrast
            if plot_indiv_sub:
                pyplot_indiv_subjects(
                    fsavg_data_diff,
                    model_name,
                    figpath,
                    save_type=save_type,
                    max_cmap_val=max_cmap_val,
                    with_rois=with_rois,
                    with_sulci=with_sulci,
                    with_labels=with_labels,
                    roi_list=roi_list,
                    roi_overlay=roi_overlay,
                    nsd_dir=nsd_dir,
                    roi_defs_dir=roi_defs_dir,
                    roi_ids=roi_ids,
                    roi_linecolor=roi_linecolor,
                    roi_linewidth=roi_linewidth,
                )
            if plot_subj_avg:
                pyplot_subj_avg(
                    fsavg_data_diff,
                    model_name,
                    figpath,
                    sig_mask='fdr_bh',
                    save_type=save_type,
                    max_cmap_val=max_cmap_val,
                    with_rois=with_rois,
                    with_sulci=with_sulci,
                    with_labels=with_labels,
                    roi_list=roi_list,
                    roi_overlay=roi_overlay,
                    nsd_dir=nsd_dir,
                    roi_defs_dir=roi_defs_dir,
                    roi_ids=roi_ids,
                    roi_linecolor=roi_linecolor,
                    roi_linewidth=roi_linewidth,
                )
