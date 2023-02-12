import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt, colors


def show_bar_value(rects, val_fmt='', ax=None):
    """
    show bars' value on the figure automatically
    Reference: https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars

    :param rects:
        bars in the matplotlib ax
    :param val_fmt: str
        value format, used to control the value's visualization format
    :param ax:
        axis of current plot
    """
    if ax is None:
        ax = plt.gca()
    for rect in rects:
        value = rect.get_height()
        label = '{0:{1}}'.format(value, val_fmt)
        if value < 0:
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    value, label, ha='center', va='top')
        else:
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    value, label, ha='center', va='bottom')


def auto_bar_width(x, item_num=1):
    """
    decide bar width automatically according to the length and interval of x indices.

    :param x: 1-D sequence
        x indices in the matplotlib ax
    :param item_num: integer
        the number of items for plots

    :return width: float
        bar width
    """
    length = len(x)
    bar_num = length * item_num
    if length > 1:
        interval = x[1] - x[0]
        width = (length - 1.0) * interval / bar_num
    else:
        width = 0.1

    return width


def imshow(X, xlabel='', ylabel='', cmap=None, cbar_label=None,
           xticklabels=None, yticklabels=None, output=None):
    """
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    """

    fig, ax = plt.subplots()
    im = ax.imshow(X, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if cbar_label is not None:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    if xticklabels is not None:
        ax.set_xticks(np.arange(X.shape[1]))
        ax.set_xticklabels(xticklabels)
        plt.setp(ax.get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')

    if yticklabels is not None:
        ax.set_yticks(np.arange(X.shape[0]))
        ax.set_yticklabels(yticklabels)

    fig.tight_layout()

    if output is not None:
        plt.savefig(output)


def check_format_y_bar_line(y):
    """
    检查和整理用于画bar或line图的y数据
    最后返回被整理成标准格式的y，即a list of 2D arrays

    Args:
        y (ndarray | list): 画bar或line图用的y数据
            必须是1/2D的array或是一列表的1/2D的array
            如果是列表，则把其中的每个array画到不同的ax里
            所有的1D array会被reshape成只有1行的2D array
            2D array的不同行是画在同一个ax里的不同item
    """
    error_info = 'y must be a 1/2D array or a list of 1/2D arrays'
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = [np.expand_dims(y, 0)]
        elif y.ndim == 2:
            y = [y]
        else:
            raise ValueError(error_info)
    elif isinstance(y, list):
        for i, e in enumerate(y):
            assert isinstance(e, np.ndarray), error_info
            if e.ndim == 1:
                y[i] = np.expand_dims(e, 0)
            elif e.ndim == 2:
                pass
            else:
                raise ValueError(error_info)
    else:
        raise TypeError(error_info)

    return y


def check_format_yerr_bar_line(yerr, n_ax, n_items, x_lengths):
    """
    检查和整理用于画bar或line图的yerr数据
    最后返回被整理成标准格式的yerr，详见yerr参数的描述

    Args:
        yerr (None | ndarray | list): 画bar或line图用的yerr数据
            如果是None, 则构建元素全是None的2层嵌套列表：
                第一层的长度为n_ax, 第二层各列表的长度和n_items里的值一一对应
            如果是array，只能是1D或2D，n_ax必须是1
                如果是1D array，则n_item必须是1，长度为x_length
                如果是2D array，形状必须是(n_item, x_length)
                最终构造成[(n_item, x_length)]
            如果是列表，则长度为n_ax，其中的元素只能是None, 1/2D array或列表：
                如果是None，则改造为全是None的列表，长度等于n_item；
                如果是1D array, 则对应的n_item必须是1，长度为对应的x_length
                    最终会变形成(1, x_length)
                如果是2D array, 形状必须是对应的(n_item, x_length)
                如果是列表，其中的元素只能是None或1D array：
                    如果是1D array，长度必须是对应的x_length
        n_ax (int): ax的数量
        n_items (integers): len(n_items) == n_ax
            各ax中item的数量
        x_lengths (integers): len(x_lengths) == n_ax
            各ax中某个item的数据长度
    """
    assert n_ax == len(n_items)
    assert n_ax == len(x_lengths)
    error_info = 'yerr does not meet specifications. '\
        'Please read docstring carefully!'
    if yerr is None:
        yerr = [[None] * i for i in n_items]
    elif isinstance(yerr, np.ndarray):
        assert n_ax == 1, error_info
        if yerr.ndim == 1:
            assert n_items[0] == 1 and len(yerr) == x_lengths[0], error_info
            yerr = [np.expand_dims(yerr, 0)]
        elif yerr.ndim == 2:
            assert yerr.shape == (n_items[0], x_lengths[0]), error_info
            yerr = [yerr]
        else:
            raise ValueError(error_info)
    elif isinstance(yerr, list):
        assert n_ax == len(yerr), error_info
        for i1, e1 in enumerate(yerr):
            if e1 is None:
                yerr[i1] = [None] * n_items[i1]
            elif isinstance(e1, np.ndarray):
                if e1.ndim == 1:
                    assert n_items[i1] == 1 and len(e1) == x_lengths[i1],\
                        error_info
                    yerr[i1] = np.expand_dims(e1, 0)
                elif e1.ndim == 2:
                    assert e1.shape == (n_items[i1], x_lengths[i1]), error_info
                else:
                    raise ValueError(error_info)
            elif isinstance(e1, list):
                for e2 in e1:
                    if e2 is None:
                        pass
                    elif isinstance(e2, np.ndarray):
                        assert e2.shape == (x_lengths[i1],), error_info
                    else:
                        raise TypeError(error_info)
            else:
                raise TypeError(error_info)
    else:
        raise TypeError(error_info)

    return yerr


def check_format_x_bar_line(x, n_ax, x_lengths):
    """
    检查和整理用于画bar或line图的x数据
    最后返回被整理成标准格式的x，详见x参数的描述

    Args:
        x (None | 1D array | list): 画bar或line图用的x数据
            如果是None, 则构建长度为n_ax的列表：
                各元素是np.arange(x_length)
            如果是1D array, n_ax必须是1，长度必须是x_length
                最终构造成[(x_length,)]
            如果是列表，则长度为n_ax，其中的元素只能是None或1D array：
                如果是None, 则改成np.arange(x_length)
                如果是1D array, 长度为对应的x_length
        n_ax (int): ax的数量
        x_lengths (integers): len(x_lengths) == n_ax
            各ax中某个item的数据长度
    """
    assert n_ax == len(x_lengths)
    error_info = 'x does not meet specifications. '\
        'Please read docstring carefully!'
    if x is None:
        x = [np.arange(i) for i in x_lengths]
    elif isinstance(x, np.ndarray):
        assert n_ax == 1 and x.shape == (x_lengths[0],), error_info
        x = [x]
    elif isinstance(x, list):
        assert n_ax == len(x), error_info
        for i1, e1 in enumerate(x):
            if e1 is None:
                x[i1] = np.arange(x_lengths[i1])
            elif isinstance(e1, np.ndarray):
                assert e1.shape == (x_lengths[i1],), error_info
            else:
                raise TypeError(error_info)
    else:
        raise TypeError(error_info)

    return x


def check_format_item_attr_bar_line(data, n_ax, n_items):
    """
    检查和整理用于画bar或line图的时候，指定item属性（label，color等）的数据
    最后返回被整理成标准格式的数据，详见data参数的描述

    Args:
        data (None | tuple | list):
            如果是None, 则构建一个长度为n_ax的列表，每个元素是一个元组，
                每个元组内的元素是对应n_item数量的None
            如果是tuple, n_ax必须是1，长度必须是n_item
                最终构造成[data]
            如果是list，则长度为n_ax，其中的元素只能是None或tuple：
                如果是None, 则改造成长度为对应n_item的值全是None元组；
                如果是tuple, 则长度为对应的n_item。
        n_ax (int): ax的数量
        n_items (integers): len(n_items) == n_ax
            各ax中item的数量
    """
    assert n_ax == len(n_items)
    error_info = 'data does not meet specifications. '\
        'Please read docstring carefully!'
    if data is None:
        data = [(None,) * i for i in n_items]
    elif isinstance(data, tuple):
        assert n_ax == 1 and len(data) == n_items[0], error_info
        data = [data]
    elif isinstance(data, list):
        assert n_ax == len(data), error_info
        for i1, e1 in enumerate(data):
            if e1 is None:
                data[i1] = (None,) * n_items[i1]
            elif isinstance(e1, tuple):
                assert len(e1) == n_items[i1], error_info
            else:
                raise TypeError(error_info)
    else:
        raise TypeError(error_info)

    return data


def check_format_tick_lim(data, n_ax):
    """
    检查和整理在调整ax时的xtick[label], ytick[label], xlim, ylim数据
    最后返回被整理成标准格式的数据，详见data参数的描述

    Args:
        data (None | 1D array | tuple | list):
            如果是None, 1D array或tuple, 则构建一个长度为n_ax的列表, 每个元素都是data
            如果是list，则长度为n_ax，其中的元素只能是None, 1D array或tuple
        n_ax (int): ax的数量
    """
    error_info = 'data does not meet specifications. '\
        'Please read docstring carefully!'
    if data is None or isinstance(data, tuple):
        data = [data] * n_ax
    elif isinstance(data, np.ndarray):
        assert data.ndim == 1, error_info
        data = [data] * n_ax
    elif isinstance(data, list):
        assert n_ax == len(data), error_info
        for e1 in data:
            if e1 is None or isinstance(e1, tuple):
                pass
            elif isinstance(e1, np.ndarray):
                assert e1.ndim == 1, error_info
            else:
                raise TypeError(error_info)
    else:
        raise TypeError(error_info)

    return data


def prepare_y_bar_line(data, key_groups):
    """
    依据key_groups从一个类似字典的数据结构中取出数据，
    并组织成满足画bar/line的y或yerr的格式

    Args:
        data (dict-like): key-value pairs
            各value是array-like的，彼此形状相同
        key_groups (sequence): 2层嵌套的序列
            第1层包含的各个子序列就是各组，子序列中的元素就是key
    """
    out_data = []
    for keys in key_groups:
        arr = [data[k] for k in keys]
        out_data.append(np.array(arr))
    return out_data


def plot_axes(fig, axes, n_ax, xlabel=None, xlim=None, xtick=None, xticklabel=None,
              rotate_xticklabel=False, ylabel=None, ylim=None, ytick=None, yticklabel=None,
              title=None, mode='show'):
    """
    设置axes的非数据部分

    Args:
        fig (Figure): matplotlib画布
        axes (2D array): 其中的值是matplotlib的axis
        n_ax (int): 被使用的axis数量，以行优先的顺序和axes内容对应
        xlabel (str | strings, optional): Defaults to None.
            If None, 什么都不做；
            If str, 为底部axis添加xlabel
            If strings, 为各axis添加xlabel
        xlim (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        xtick (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        xticklabel (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        rotate_xticklabel (bool, optional): Defaults to False.
        ylabel (str | strings, optional): Defaults to None.
            If None, 什么都不做；
            If str, 为左侧axis添加ylabel
            If strings, 为各axis添加ylabel
        ylim (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        ytick (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        yticklabel (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        title (str | strings, optional): Defaults to None.
            If None, 什么都不做；
            If str, 为顶部中间的axis添加title
            If strings, 为各axis添加title
        mode (str, optional): Defaults to 'show'.
            If 'show', 可视化画布
            elif 'go on', 返回fig, axes, n_ax给后续加工步骤
            else, 默认是图片路径，将画布存到硬盘。

    Returns:
        [type]: [description]
    """
    xlim = check_format_tick_lim(xlim, n_ax)
    xtick = check_format_tick_lim(xtick, n_ax)
    xticklabel = check_format_tick_lim(xticklabel, n_ax)
    ylim = check_format_tick_lim(ylim, n_ax)
    ytick = check_format_tick_lim(ytick, n_ax)
    yticklabel = check_format_tick_lim(yticklabel, n_ax)

    n_row, n_col = axes.shape
    assert n_ax <= n_row * n_col

    max_row_idx = int((n_ax - 1) / n_col)
    for ax_idx in range(n_ax):
        row_idx = int(ax_idx / n_col)
        col_idx = ax_idx % n_col
        ax = axes[row_idx, col_idx]

        if xlabel is None:
            pass
        elif isinstance(xlabel, str):
            if row_idx == max_row_idx:
                ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xlabel[ax_idx])

        if ylabel is None:
            pass
        elif isinstance(ylabel, str):
            if col_idx == 0:
                ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(ylabel[ax_idx])

        if title is None:
            pass
        elif isinstance(title, str):
            if row_idx == 0 and col_idx == int(n_col/2):
                ax.set_title(title)
        else:
            ax.set_title(title[ax_idx])

        if xlim[ax_idx] is not None:
            ax.set_xlim(*xlim[ax_idx])
        if xticklabel[ax_idx] is not None:
            assert len(xtick[ax_idx]) == len(xticklabel[ax_idx])
            ax.set_xticks(xtick[ax_idx])
            ax.set_xticklabels(xticklabel[ax_idx])

        if ylim[ax_idx] is not None:
            ax.set_ylim(*ylim[ax_idx])
        if yticklabel[ax_idx] is not None:
            assert len(ytick[ax_idx]) == len(yticklabel[ax_idx])
            ax.set_yticks(ytick[ax_idx])
            ax.set_yticklabels(yticklabel[ax_idx])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if rotate_xticklabel:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

    fig.tight_layout()
    if mode == 'show':
        fig.show()
    elif mode == 'go on':
        return fig, axes, n_ax
    else:
        fig.savefig(mode)


def plot_bar(y, n_row=1, n_col=1, figsize=None, yerr=None, x=None, width=None,
             label=None, color=None, fc_ec_flag=False, fc=None, ec=None, show_height=None,
             xlabel=None, xticklabel=None, rotate_xticklabel=False, ylabel=None, ylim=None,
             title=None, mode='show'):
    """
    基本上满足单纵轴所有常用bar图的绘制了。

    Args:
        y (1/2D array | list): 详见check_format_y_bar_line
        n_row (int, optional): axes行数. Defaults to 1.
        n_col (int, optional): axes列数. Defaults to 1.
        figsize (tuple, optional): 画布大小 宽x高. Defaults to None.
        yerr (1/2D array | list, optional): Defaults to None.
            详见check_format_yerr_bar_line
        x (1D array | list, optional): Defaults to None.
            详见check_format_x_bar_line
        width (float, optional): bar的宽度. Defaults to None.
        label (tuple | list, optional): Defaults to None.
            详见check_format_item_attr_bar_line
        color (tuple | list, optional): Defaults to None.
            详见check_format_item_attr_bar_line
        fc_ec_flag (bool, optional): Defaults to False
            If True, use fc and ec parameters
                fc和ec设置为None时，bar不会自动变颜色
            If False, use color parameter
        fc (tuple | list, optional): Defaults to None.
            详见check_format_item_attr_bar_line
        ec (tuple | list, optional): Defaults to None.
            详见check_format_item_attr_bar_line
        show_height (str, optional): Defaults to None.
            详见show_bar_value的val_fmt参数说明
        xlabel (str | strings, optional): 详见plot_axes. Defaults to None.
        xticklabel (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        rotate_xticklabel (bool, optional): Defaults to False.
        ylabel (str | strings, optional): 详见plot_axes. Defaults to None.
        ylim (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        title (str | strings, optional): 详见plot_axes. Defaults to None.
        mode (str, optional): 详见plot_axes. Defaults to 'show'.
    """
    # check data
    y = check_format_y_bar_line(y)
    n_ax = len(y)
    n_items = [i.shape[0] for i in y]
    x_lengths = [i.shape[1] for i in y]

    assert n_ax <= n_row * n_col

    yerr = check_format_yerr_bar_line(yerr, n_ax, n_items, x_lengths)

    x = check_format_x_bar_line(x, n_ax, x_lengths)

    label = check_format_item_attr_bar_line(label, n_ax, n_items)
    if fc_ec_flag:
        fc = check_format_item_attr_bar_line(fc, n_ax, n_items)
        ec = check_format_item_attr_bar_line(ec, n_ax, n_items)
    else:
        color = check_format_item_attr_bar_line(color, n_ax, n_items)

    # prepare figure and axes
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row == 1 and n_col == 1:
        axes = np.array([[axes]])
    elif axes.shape != (n_row, n_col):
        axes = axes.reshape((n_row, n_col))

    # plot on axes
    for ax_idx, ax_data in enumerate(y):
        row_idx = int(ax_idx / n_col)
        col_idx = ax_idx % n_col
        ax = axes[row_idx, col_idx]
        if width is None:
            width = auto_bar_width(x[ax_idx], n_items[ax_idx])
        offset = -(n_items[ax_idx] - 1) / 2
        for item_idx in range(n_items[ax_idx]):
            if fc_ec_flag:
                rects = ax.bar(
                    x[ax_idx] + width*offset, ax_data[item_idx], width,
                    yerr=yerr[ax_idx][item_idx], label=label[ax_idx][item_idx],
                    fc=fc[ax_idx][item_idx], ec=ec[ax_idx][item_idx])
            else:
                rects = ax.bar(
                    x[ax_idx] + width*offset, ax_data[item_idx], width,
                    yerr=yerr[ax_idx][item_idx], label=label[ax_idx][item_idx],
                    color=color[ax_idx][item_idx])

            if show_height is not None:
                show_bar_value(rects, show_height, ax)
            offset += 1
        if np.any([i is not None for i in label[ax_idx]]):
            ax.legend()

    # adjust axes
    plot_axes(fig, axes, n_ax, xlabel=xlabel, xtick=x, xticklabel=xticklabel,
              rotate_xticklabel=rotate_xticklabel, ylabel=ylabel, ylim=ylim,
              title=title, mode=mode)


def plot_line(y, n_row=1, n_col=1, figsize=None, yerr=None, x=None,
              label=None, color=None, xlabel=None, xtick=None, xticklabel=None,
              rotate_xticklabel=False, ylabel=None, ylim=None, title=None, mode='show'):
    """
    基本上满足单纵轴所有常用line图的绘制了。

    Args:
        y (1/2D array | list): 详见check_format_y_bar_line
        n_row (int, optional): axes行数. Defaults to 1.
        n_col (int, optional): axes列数. Defaults to 1.
        figsize (tuple, optional): 画布大小 宽x高. Defaults to None.
        yerr (1/2D array | list, optional): Defaults to None.
            详见check_format_yerr_bar_line
        x (1D array | list, optional): Defaults to None.
            详见check_format_x_bar_line
        label (tuple | list, optional): Defaults to None.
            详见check_format_item_attr_bar_line
        color (tuple | list, optional): Defaults to None.
            详见check_format_item_attr_bar_line
        xlabel (str | strings, optional): 详见plot_axes. Defaults to None.
        xtick (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        xticklabel (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        rotate_xticklabel (bool, optional): Defaults to False.
        ylabel (str | strings, optional): 详见plot_axes. Defaults to None.
        ylim (1D array | tuple | list, optional): Defaults to None.
            详见check_format_tick_lim
        title (str | strings, optional): 详见plot_axes. Defaults to None.
        mode (str, optional): 详见plot_axes. Defaults to 'show'.
    """
    # check data
    y = check_format_y_bar_line(y)
    n_ax = len(y)
    n_items = [i.shape[0] for i in y]
    x_lengths = [i.shape[1] for i in y]

    assert n_ax <= n_row * n_col

    yerr = check_format_yerr_bar_line(yerr, n_ax, n_items, x_lengths)

    x = check_format_x_bar_line(x, n_ax, x_lengths)

    label = check_format_item_attr_bar_line(label, n_ax, n_items)
    color = check_format_item_attr_bar_line(color, n_ax, n_items)

    # prepare figure and axes
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row == 1 and n_col == 1:
        axes = np.array([[axes]])
    elif axes.shape != (n_row, n_col):
        axes = axes.reshape((n_row, n_col))

    # plot on axes
    for ax_idx, ax_data in enumerate(y):
        row_idx = int(ax_idx / n_col)
        col_idx = ax_idx % n_col
        ax = axes[row_idx, col_idx]
        for item_idx in range(n_items[ax_idx]):
            ax.errorbar(
                x[ax_idx], ax_data[item_idx], yerr[ax_idx][item_idx],
                label=label[ax_idx][item_idx], color=color[ax_idx][item_idx])
        if np.any([i is not None for i in label[ax_idx]]):
            ax.legend()

    # adjust axes
    if xtick is None:
        xtick = x
    plot_axes(fig, axes, n_ax, xlabel=xlabel, xtick=xtick, xticklabel=xticklabel,
              rotate_xticklabel=rotate_xticklabel, ylabel=ylabel, ylim=ylim,
              title=title, mode=mode)


def plot_polyfit(x, y, deg, scoring='r', scatter_plot=True,
                 color=None, label=None, s=None, c=None, marker=None,
                 n_sample=100, ax=None):
    """
    Parameters
    ----------
    x : ndarray, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : ndarray, shape (M,)
        y-coordinates of the M sample points ``(x[i], y[i])``.
    deg : int
        Degree of the fitting polynomial
    scoring : str
        a method used to evaluate the fit effect
    scatter_plot : bool
        If True, do scatter plot
    color : parameters for plt.plot
    label : parameters for plt.plot and plt.scatter
    s, c, marker : parameters for plt.scatter
    n_sample : positive integer
        the number of points used to plot fitted curve
    ax : matplotlib axis

    References
    ----------
    1. https://blog.csdn.net/fffsolomon/article/details/104831050
    """
    # fit and construct polynomial
    coefs = np.polyfit(x, y, deg)
    polynomial = np.poly1d(coefs)
    print('polynomial:\n', polynomial)

    # scoring
    y_pred = polynomial(x)
    if scoring == 'r':
        score = pearsonr(y, y_pred)
    elif scoring == 'r2_score':
        score = r2_score(y, y_pred)
    elif scoring == 'spearmanr':
        score = spearmanr(y, y_pred)
    else:
        raise ValueError("Not supported scoring:", scoring)
    print('\nscore:', score)

    if ax is None:
        _, ax = plt.subplots()

    # plot scatter
    if scatter_plot:
        ax.scatter(x, y, c=c, label=label, s=s, marker=marker)

    # plot fitted curve
    x_min, x_max = np.min(x), np.max(x)
    x_plot = np.linspace(x_min, x_max, n_sample)
    y_plot = polynomial(x_plot)
    ax.plot(x_plot, y_plot, color=color, label=label)

    return ax


class VlineMover(object):
    """
    Move the vertical line when left button is clicked.
    """

    def __init__(self, vline, x_round=False):
        """
        :param vline: Matplotlib Line2D
            the vertical line object
        :param x_round: bool
            If true, round the x index.
        """
        self.vline = vline
        self.x_round = x_round
        self.ax = vline.axes
        self.x = vline.get_xdata()
        self.y = vline.get_ydata()
        self.cid = vline.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.button == 1 and event.inaxes is self.ax:
            if self.x_round:
                self.x = [round(event.xdata)] * 2
            else:
                self.x = [event.xdata] * 2
            self.vline.set_data(self.x, self.y)
            self.vline.figure.canvas.draw()


class VlineMoverPlotter(object):
    """
    plot a figure with vertical line interaction
    """

    def __init__(self, nrows=1, ncols=1, sharex=False, sharey=False,
                 squeese=True, subplot_kw=None, gridspec_kw=None, **fig_kw):

        self.figure, self.axes = plt.subplots(nrows, ncols, sharex, sharey,
                                              squeese, subplot_kw, gridspec_kw, **fig_kw)
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        self.figure.canvas.mpl_connect('button_press_event', self._on_clicked)

        self.axes_twin = np.zeros_like(self.axes, dtype=np.object)
        self.vline_movers = np.zeros_like(self.axes)

    def add_twinx(self, idx=0, r_idx=0, c_idx=0):
        """
        create and save twin axis for self.axes[idx] or self.axes[r_idx, c_idx]

        :param idx: integer
            The index of the self.axes. (Only used when self.axes.ndim == 1)
        :param r_idx: integer
            The row index of the self.axes. (Only used when self.axes.ndim == 2)
        :param c_idx: integer
            The column index of the self.axes. (Only used when self.axes.ndim == 2)
        :return:
        """
        if self.axes.ndim == 1:
            self.axes_twin[idx] = self.axes[idx].twinx()
        elif self.axes.ndim == 2:
            self.axes_twin[r_idx, c_idx] = self.axes[r_idx, c_idx].twinx()

    def add_vline_mover(self, idx=0, r_idx=0, c_idx=0, vline_idx=0, x_round=False):
        """
        add vline mover for each ax

        :param idx: integer
            The index of the self.axes. (Only used when self.axes.ndim == 1)
        :param r_idx: integer
            The row index of the self.axes. (Only used when self.axes.ndim == 2)
        :param c_idx: integer
            The column index of the self.axes. (Only used when self.axes.ndim == 2)
        :param vline_idx: integer
            A index used to initialize the vertical line's position
        :param x_round: bool
            If true, round the x index.
        :return:
        """
        if self.axes.ndim == 1:
            if self.axes_twin[idx] == 0:
                self.vline_movers[idx] = VlineMover(self.axes[idx].axvline(vline_idx), x_round)
            else:
                self.vline_movers[idx] = VlineMover(self.axes_twin[idx].axvline(vline_idx), x_round)
        elif self.axes.ndim == 2:
            if self.axes_twin[r_idx, c_idx] == 0:
                self.vline_movers[r_idx, c_idx] = VlineMover(self.axes[r_idx, c_idx].axvline(vline_idx), x_round)
            else:
                self.vline_movers[r_idx, c_idx] = VlineMover(self.axes_twin[r_idx, c_idx].axvline(vline_idx), x_round)

    def _on_clicked(self, event):
        pass


class MidpointNormalize(colors.Normalize):
    # https://matplotlib.org/stable/gallery/userdemo/colormap_normalizations.html
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
