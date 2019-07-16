import numpy as np
import matplotlib
import pylab
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt

FULL_LABELS = ['Frontal', 'Parietal', 'Temporal', 'Occipital', 'Subcortical',
               'Occipital', 'Temporal', 'Parietal', 'Frontal']
FULL_LABELS_IT = ['Frontal', 'Parietal', 'Temporal', 'Occipital',
                  'Subcortical/Infratentorial', 'Occipital', 'Temporal',
                  'Parietal', 'Frontal']
LABELS_LR = ['Frontal', 'Temporal', 'Subcortical', 'Occipital', 'Parietal']


def rtext(line, x, y, s, **kwargs):
    from scipy.optimize import curve_fit
    xdata, ydata = line.get_data()
    xdata = xdata[::40]
    ydata = ydata[::40]
    dist = np.sqrt((x-xdata)**2 + (y-ydata)**2)
    dmin = dist.min()
    TOL_to_avoid_rotation = 0.3
    if dmin > TOL_to_avoid_rotation:
        r = 0.
    else:
        index = dist.argmin()
        xs = xdata[[index-2, index-1, index, index+1, index+2]]
        ys = ydata[[index-2, index-1, index, index+1, index+2]]
        
        def f(x, a0, a1, a2, a3):
            return a0 + a1*x + a2*x**2 + a3*x**3
        popt, pcov = curve_fit(f, xs, ys, p0=(1, 1, 1, 1))
        a0, a1, a2, a3 = popt
        ax = pylab.gca()
        derivative = (a1 + 2*a2*x + 3*a3*x**2)
        derivative /= ax.get_data_ratio()
        r = np.arctan( derivative )
    return pylab.text(x, y, s, rotation=np.rad2deg(r), **kwargs)


def prepare_data_fromagglo(data, num_layers=4, type_prepa="full", 
                           corr_it=False):
    data_init = None
    if type_prepa == "full":
        if corr_it:
            data_init = data[1:9*num_layers+1]
        else:
            begin_full_bgit = 10*num_layers+1+10+num_layers+4*num_layers
            data_init = np.concatenate((data[1:8*num_layers], data[
                                      begin_full_bgit:begin_full_bgit  
                                                      + num_layers + 1]), 0)
        preparation = [1, 3, 7, 5, 8, 4, 6, 2, 0]
    elif type_prepa == 'lr':
        count_before = 10*num_layers+1+10+num_layers
        data_init = data[count_before:count_before+4*num_layers]
        if corr_it:
            data_init = np.concatenate((data_init, data[1+8*num_layers:
            1 + 9 * num_layers+1]), 0)
        else:
            data_init = np.concatenate((data_init, data[
                                    count_before+4*num_layers:count_before + 
                                                              5*num_layers]), 0)
        preparation = [0, 3, 4, 2, 1]
    data_res = np.reshape(data_init, [-1, num_layers]).T
    data_reshuffled = data_res[:, preparation]
    data_prepared = np.reshape(data_reshuffled, [1, -1])
    return data_prepared


def prepare_data_bullseye(filename, num_layers=4, corr_it=False):
    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
   # FL FR PL PR OL OR TL TR BG IT
   # 0  1  2  3  4  5  6  7  8
   # FR PR TR OR BGIT OL TL PL FL

    v_prob = np.zeros([9*num_layers, 1])
    v_reg = np.zeros([9*num_layers, 1])
    preparation = [1, 3, 7, 5, 8, 4, 6, 2, 0]
    preparation *= num_layers
    preparation_init = preparation
    for l in range(1, num_layers):
        preparation += list(np.asarray(preparation_init)+l)
    # preparation = preparation + list(np.asarray(preparation)+1) + list(
    #     np.asarray(preparation)+2) +list(np.asarray(preparation)+3)
    for i in range(0,len(preparation)):
        string_split = content[preparation[i]].split(' ')
        v_prob[i] = float(string_split[0])
        v_reg[i] = float(string_split[2])
        if np.floor_divide(i,num_layers) == 8 and not corr_it:
            v_prob[i] += float(content[9*num_layers].split(' ')[0])
            v_reg[i] += float(content[9 * num_layers].split(' ')[2])
    string_tot = content[10*num_layers].split(' ')
    VPerc = np.divide(v_prob, v_reg)
    VDist = np.divide(v_prob, float(string_tot[0]))
    return VPerc, VDist


def read_ls_create_agglo(filename, num_layers=4, corr_it=False):
    with open(filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    v_prob = []
    v_reg = []
    for i in range(0, len(content)-1):
        string_split = content[i].split(' ')
        v_prob.append(float(string_split[0]))
        v_reg.append(float(string_split[2]))
    v_prob_tot=float(content[-1].split(' ')[0])
    v_reg_tot = float(content[-1].split(' ')[2])
    v_prob_res = np.reshape(v_prob, [num_layers,-1])
    v_reg_res = np.reshape(v_reg, [num_layers, -1])
    v_prob_layers = np.sum(v_prob_res,1)
    v_prob_lobes = np.sum(v_prob_res,0)
    v_prob_combo_lr = v_prob_res[:,0:-1:2] + v_prob_res[:, 1::2]
    v_prob_lobes_lr = np.sum(v_prob_combo_lr, 0)
    v_reg_layers = np.sum(v_reg_res, 1)
    v_reg_lobes = np.sum(v_reg_res, 0)
    v_reg_combo_lr = v_reg_res[:, 0:-1:2] + v_reg_res[:, 1::2]
    v_reg_lobes_lr = np.sum(v_reg_combo_lr, 0)
    les_fin = np.concatenate(([v_prob_tot], v_prob, v_prob_layers, v_prob_lobes,
                        np.reshape(v_prob_combo_lr, -1), v_prob_lobes_lr), 0)
    reg_fin = np.concatenate(([v_reg_tot], v_reg, v_reg_layers, v_reg_lobes,
                           np.reshape(
        v_reg_combo_lr,-1),v_reg_lobes_lr), 0)
    freq_fin = les_fin/reg_fin
    dist_fin = les_fin/v_prob_tot
    return les_fin, reg_fin, freq_fin, dist_fin


def agglo_ls_without_speclobe(les, reg, num_layers=4, lobe_remove=None):
    les_init = np.reshape(les[1:num_layers*10+1], [num_layers, -1])
    reg_init = np.reshape(reg[1:num_layers*10+1], [num_layers, -1])
    remove_idx1 = [2*l for l in lobe_remove if l<4]
    remove_idx2 = [2*l+1 for l in lobe_remove if l<4]
    remove_tot = remove_idx1 + remove_idx2
    for l in lobe_remove:
        if l>=4:
            remove_tot.append(l+5)
    idx = [i for i in range(0,10) if i not in remove_tot]
    les_without = les_init[:,idx]
    reg_without = reg_init[:,idx]

    v_prob_layers = np.sum(les_without, 1)
    v_prob_lobes = np.sum(reg_without, 0)
    v_prob_combo_lr = les_without[:, 0:-1:2] + les_without[:, 1::2]
    v_prob_lobes_lr = np.sum(v_prob_combo_lr, 0)
    v_reg_layers = np.sum(reg_without, 1)
    v_reg_lobes = np.sum(reg_without, 0)
    v_reg_combo_lr = reg_without[:, 0:-1:2] + reg_without[:, 1::2]
    v_reg_lobes_lr = np.sum(v_reg_combo_lr, 0)
    les_fin = np.concatenate(([np.sum(les_without)],np.reshape(les_without,
                                                              [-1]),
                                          v_prob_layers,
                              v_prob_lobes,
                              np.reshape(v_prob_combo_lr, -1), v_prob_lobes_lr),
                             0)
    reg_fin = np.concatenate(([np.sum(reg_without)], np.reshape(reg_without,
                                                              [-1]),
                              v_reg_layers,
                              v_reg_lobes,
                              np.reshape(
                                  v_reg_combo_lr, -1), v_reg_lobes_lr), 0)
    freq_fin = les_fin / reg_fin
    dist_fin = les_fin / np.sum(les_without)
    return les_fin, reg_fin, freq_fin, dist_fin



def create_bullseye_plot(data, color, num_layers=4, num_lobes=9, vmin=0,
                         vmax=1, labels=FULL_LABELS, thr=None):
    n_layer_steps = num_layers + 1
    n_layer_steps_comp = n_layer_steps * 1j
    theta, r = np.mgrid[0:2*np.pi:(num_lobes*40+1)*1j, 0.2:1:n_layer_steps_comp]
    print(theta.shape)
    z = np.tile(data, [40, 1]).T

    z = z.reshape([num_layers, num_lobes*40])

    z_new = np.concatenate((z, -1*np.ones([1,num_lobes*40])),
                                      axis=0).T
    z_new = np.concatenate((z_new, -1*np.ones([1, n_layer_steps])), axis=0)
    z_new = z_new.reshape(theta.shape)

    print(z_new.shape)
    rot = Affine2D().rotate_deg(30)

    colormap = plt.get_cmap(color)
    colormap.set_bad('grey')
    if thr is not None:
        z_new = np.where(z_new < thr, -1000*np.ones_like(z_new),
                         z_new )
    z_new = np.ma.masked_values(z_new, -1000)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection='polar',
                                                    transform=rot))
    ax.pcolormesh(theta, r, z_new,clip_on=True, cmap=colormap,
                  edgecolors='face', antialiased=True, vmin=vmin, vmax=vmax)
    for i in range(0, n_layer_steps):
        ax.plot(theta.T[0], np.repeat(r[0][i], theta.T[0].size), '-',
                color=[0.5, 0.5, 0.5], lw=1)
    for i in range(num_lobes):
        theta_i = i * 360/num_lobes * np.pi / 180
        ax.plot([theta_i, theta_i], [r[0][0], 1], '-', color=[0.5, 0.5, 0.5], 
                lw=1)

    ax.set_theta_zero_location("N")
    xT = np.arange(180/num_lobes*np.pi/180, 2*np.pi, 
                   step=360/num_lobes*np.pi/180)
    xL = labels
    xdata = xT
    ydata = np.asarray([1.1, ]*len(xT))

    for x, y, l in zip(xdata, ydata, xL):
        if x > np.pi/2 and x < 3*np.pi/2:
            x_rot = x+np.pi
        else:
            x_rot = x
        pylab.text(x, y, l, rotation=np.rad2deg(x_rot), fontsize=10,
                   horizontalalignment='center', verticalalignment='center')
    # for (t,l) in zip(xT,xL):
    #     plt.xticks(t, l, y=0.11, rotation=t)

    axl = fig.add_axes([0.87, 0.1, 0.03, 0.8])
    cb1 = matplotlib.colorbar.ColorbarBase(axl, cmap=colormap, norm=norm,
                                           orientation='vertical')

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return plt.gcf()

