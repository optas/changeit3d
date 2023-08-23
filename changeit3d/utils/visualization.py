"""
Utilities to visualize 2D & 3D data.

(c) 2021 Panos Achlioptas (https://optas.github.io)
"""

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from mpl_toolkits.mplot3d import axes3d, Axes3D # nofa: #401 need to call to work matplotlib 3d
from ..in_out.pointcloud import swap_axes_of_pointcloud


def plot_3d_point_cloud(pc, show=True, show_axis=True, in_u_sphere=True, marker='.', s=8, alpha=.8, set_lim=1.1,
                        figsize=(5, 5), elev=10, azim=245, axis=None, title=None, visualization_pc_axis=(0,1,2), *args, **kwargs):
    """Plot a 3d point-cloud via matplotlib.
    :param pc: N x 3 numpy array storing the x, y, z coordinates of a 3D pointcloud with N points.
    Students Note you can use the default other parameters, or explore their effect for better visualization.
    """

    if visualization_pc_axis != (0, 1, 2):
        pc = swap_axes_of_pointcloud(pc.copy(), visualization_pc_axis)

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        set_lim = 1

    if set_lim is not None:
        ax.set_xlim3d(-set_lim, set_lim)
        ax.set_ylim3d(-set_lim, set_lim)
        ax.set_zlim3d(-set_lim, set_lim)

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig


def visualize_point_clouds_3d_v2(pcl_lst, title_lst=None, vis_axis_order=[0, 2, 1], fig_title=None):
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)
    
    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    if fig_title is not None:
        plt.title(fig_title)
        
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)       
        ax1.scatter(pts[:, vis_axis_order[0]], pts[:, vis_axis_order[1]], pts[:, vis_axis_order[2]], s=2)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    res = Image.fromarray(res[:3].transpose(1,2,0))
    return res


def stack_images_horizontally(file_names, size=None, save_file=None, input_is_image=False):
    """ Opens the images corresponding to file_names and
    creates a new image stacking them horizontally.
    """

    if input_is_image:
        images = file_names
    else:
        if size is not None:
            def opener(im):
                return Image.open(im).resize(size, Image.ANTIALIAS)

            images = list(map(opener, file_names))
        else:
            images = list(map(Image.open, file_names))

    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_file is not None:
        new_im.save(save_file)
    return new_im


def stack_images_vertically(file_names, size=None, save_file=None, input_is_image=False):
    """ Opens the images corresponding to file_names and
    creates a new image stacking them vertically.
    """

    if input_is_image:
        images = file_names
    else:
        if size is not None:
            def opener(im):
                return Image.open(im).resize(size, Image.ANTIALIAS)

            images = list(map(opener, file_names))
        else:
            images = list(map(Image.open, file_names))

    widths, heights = list(zip(*(i.size for i in images)))
    total_width = max(widths)
    max_height = sum(heights)
    new_im = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (0, x_offset))
        x_offset += im.size[1]
    if save_file is not None:
        new_im.save(save_file)
    return new_im


def visualize_contrastive_triplet_in_df(df, loc, top_img_dir, class_name="chair",
                                        image_base_name='image_p020_t337_r005.png',
                                        utterance_original=False):
    image_files = []
    for x in ['a', 'b', 'c']:
        x = class_name + '_' +x
        img_file = osp.join(top_img_dir, df.iloc[loc][x], image_base_name)
        image_files.append(img_file)

    if 'target_chair' in df.columns:
        print(df.iloc[loc]['target_chair'])


    print(df.iloc[loc]['utterance'])
    if utterance_original:
        print(df.iloc[loc]['utterance_original'])

    if 'correct' in df.columns:
        print(df.iloc[loc]['correct'])
    return stack_images_horizontally(image_files)


def visualize_src_trg_df_at_loc_i(df, loc_i, top_img_dir, size=None, max_utters=5, img_ending='.png',
                                  print_utters=True):
    row = df.loc[loc_i]
    imgs_files = []
    for x in ['source', 'target']:
        dataset_name = row[f'{x}_dataset']
        obj_class = row[f'{x}_object_class']
        model_name = row[f'{x}_model_name']
        imgs_files.append(osp.join(top_img_dir, obj_class, dataset_name, model_name + img_ending))

    display(stack_images_horizontally(imgs_files, size=size))

    if 'assignmentid' in df.columns:
        print(row['assignmentid'])

    if print_utters:
        if 'utterance0' in df.columns:
            utterance_columns = [f'utterance{i}' for i in range(max_utters)]
            for i, item in enumerate(row[utterance_columns]):
                print(utterance_columns[i]+':', item)
        elif 'utterance_0' in df.columns:
            utterance_columns = [f'utterance_{i}' for i in range(max_utters)]
            for i, item in enumerate(row[utterance_columns]):
                print(utterance_columns[i]+':', item)
        else:
            if 'saliency' in df.columns:
                print(f'utterance{row["saliency"]}:')

            print('Original:', row['utterance'])
            if 'utterance_spelled' in df.columns:
                print('Spelled:', row['utterance_spelled'])
