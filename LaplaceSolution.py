
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.morphology import distance_transform_edt as dist_edt
from bisect import bisect_right
from skimage.morphology import skeletonize_3d
from scipy.spatial.distance import cdist
import os
import nibabel as nib

def lesion_laplace_integral(intensities, lesion_mask, pixdim, value=10000,
                          thresh_conv=0.00005):
    #dil_mask = binary_dilation(lesion_mask, iterations=3).astype(int)
    dil_mask = lesion_mask
    skeleton = skeletonize_3d(dil_mask)
    border_lesion = dil_mask - binary_erosion(dil_mask, iterations=1)
    laplace_sol = solving_laplace_equation(dil_mask, border_lesion,
                                           skeleton, pixdim, value,
                                           thresh_conv)*dil_mask
    integral = integral_along_laplace(intensities, skeleton, laplace_sol,
                                   pixdim,
                           dil_mask, value)
    return skeleton, laplace_sol, integral


def solving_laplace_equation(mask, surface_out, surface_in, pixdim, value=10000,
                             thresh_conv=0.00005):
    old_flux = initialise_laplace(mask, surface_out, surface_in, value)
    old_energy = 0.00001
    flag_continue = True
    while(flag_continue):
        current_flux = update_flux(old_flux, mask, pixdim)
        current_flux = refine_flux(current_flux, surface_in, surface_out)
        new_energy = total_energy_calculation(current_flux, surface_out,
                                              surface_in, mask, pixdim)
        print(new_energy, np.abs(new_energy-old_energy)/old_energy)
        if(np.abs(new_energy-old_energy)/old_energy<=thresh_conv):
            flag_continue = False
        old_energy = new_energy
        old_flux = np.copy(current_flux)
    return old_flux


def update_flux(old_flux, mask, pixdim):
    shift_final = np.zeros_like(old_flux)
    sum_norm_factor = 0
    pixdim = np.asarray(pixdim)
    for x in range(-1, 2):
        for y in range(-1 + abs(x), 2-abs(x)):
            for z in range(-1 + abs(x)+abs(y),2 - abs(x) - abs(y)):
                if (abs(x)+abs(y)+abs(z))>0:
                    trans = [x, y, z]
                    distance = np.sum(np.square(trans*pixdim))
                    normalisation = np.sqrt(distance)
                    shift_temp = np.roll(old_flux, trans, axis=[0, 1, 2])
                    sum_norm_factor += 1/normalisation
                    shift_final += shift_temp * 1/normalisation
    return shift_final / sum_norm_factor


def refine_flux(current_flux, surface_in, surface_out,
                value=10000):
    current_flux = np.where(surface_in, np.zeros_like(current_flux),
                            current_flux)
    current_flux = np.where(surface_out, value * np.ones_like(current_flux),
                            current_flux)
    current_flux = np.where(current_flux > value, np.ones_like(current_flux)
                            * value, current_flux)
    return current_flux


def total_energy_calculation(current_flux, surface_out, surface_in, mask,
                             pixdim):
    list_grad = np.gradient(current_flux, axis=(0,1,2))
    euc_grad = np.square(list_grad[0] / pixdim[0])
    euc_grad += np.square(list_grad[1] / pixdim[1])
    euc_grad += np.square(list_grad[2] / pixdim[2])
    mask_minus = mask - surface_out -surface_in
    euc_tosum = np.where(mask_minus>0, np.sqrt(euc_grad), np.zeros_like(
        mask_minus))
    return np.sum(euc_tosum)


def initialise_laplace(mask, surface_out, surface_in, value):
    init = np.ones_like(mask) * -1
    init = np.where(mask, value/2 * np.ones_like(init), init)
    init = np.where(surface_out, value * np.ones_like(init), init)
    init = np.where(surface_in, np.zeros_like(init), init)
    return init


def normalised_laplace_length(laplace_sol, object_in, object_out, pixdim, mask):
    dist_in = initial_distance(object_in, pixdim, mask, 1)
    dist_out = initial_distance(object_out, pixdim, mask, -1)
    length_solving_laplace(dist_in, object_in, laplace_sol, pixdim, mask)
    print(np.max(dist_in))
    length_solving_laplace(dist_out, object_out, laplace_sol*-1, pixdim, mask)
    print(np.max(dist_out))
    normalised = np.nan_to_num(dist_in / (dist_in + dist_out))
    return normalised


def initial_distance(object, pixdim, mask, direction):

    bin_ero = binary_erosion(object, iterations=1)
    border = object - bin_ero
    if direction == 1:
        bin_ext = binary_dilation(object)
        dil_bord = bin_ext - object
    else:
        bin_ext = binary_erosion(object, iterations=2)
        dil_bord = np.asarray(bin_ero, dtype=float) - np.asarray(bin_ext,
                                                                 dtype=float)

    distance = np.zeros_like(object)
    dist_part = np.abs(dist_edt(1-border, sampling=pixdim) * dil_bord)
    distance = np.where(dil_bord, dist_part, distance)
    return distance




def integral_along_laplace(to_integrate, object, laplace_sol, pixdim, mask,
                           value=10000, skeleton=True):
    border = object
    integral = (binary_dilation(border, iterations=1) - border) * mask * \
               to_integrate

    #integral = np.zeros_like(to_integrate)
    # normalise Laplace solution
    normalised_laplace = laplace_sol / value
    integral = np.where(normalised_laplace == 0, np.zeros_like(integral),
                        integral)
    tangent = create_tangent_laplace(normalised_laplace, pixdim, mask)

    # Create status array
    status_array = np.ones_like(mask) * -1
    status_array = np.where(mask > 0, 2 * np.ones_like(mask), status_array)
    status_array = np.where(np.logical_or(border, normalised_laplace==0),
                                          np.zeros_like(
        mask), status_array)
    status_array = np.where(integral > 0, np.ones_like(mask), status_array)

    list_status_init = np.asarray(np.where(status_array == 1)).T
    list_int = [normalised_laplace[s[0], s[1], s[2]] for s in list_status_init]
    ind_sort = np.argsort(list_int)
    list_status = list_status_init[ind_sort, :]
    list_int = [list_int[i] for i in ind_sort]
    while len(np.where(status_array == 2)) > 0 and \
            list_status.shape[0] > 0:

        ind_interest = list_status[0, :]
        list_status = list_status[1:, :]
        list_int = list_int[1:]
        print(np.asarray(np.where(status_array==2)).shape, integral[
            ind_interest[0], ind_interest[1], ind_interest[2]])
        #print(ind_interest)
        print(normalised_laplace[ind_interest[0],ind_interest[1],
                                 ind_interest[2]])
        status_array[ind_interest[0], ind_interest[1], ind_interest[2]] = 0
        #print(status_array[ind_interest[0], ind_interest[1], ind_interest[2]])
        for xi in range(-1, 2):
            for yi in range(-1, 2):
                for zi in range(-1, 2):
                    ind_shift = [xi+ind_interest[0], yi+ind_interest[1],
                                 zi+ind_interest[2]]
                    if abs(xi) + abs(yi) + abs(zi) == 1 and \
                            mask[ind_shift[0], ind_shift[1], ind_shift[2]] and \
                            status_array[ind_shift[0], ind_shift[1],
                                         ind_shift[2]] >1 and ind_shift[0]>0 \
                            and ind_shift[1] > 0 and ind_shift[2]>0 and \
                            ind_shift[0] < mask.shape[0] and ind_shift[1] < \
                            mask.shape[1] and ind_shift[2] < mask.shape[2]:


                        tangent_ind = tangent[ind_shift[0], ind_shift[1],
                                              ind_shift[2]]
                        normed_grad = np.sign(tangent_ind)

                        neigh_use = np.sign(tangent_ind)
                        print(tangent_ind, normed_grad, neigh_use)
                        list_neighbours_valid = []
                        for d in range(0, 3):
                            shift_new = [0, 0, 0]
                            shift_new[d] += neigh_use[d]
                            ind_shift_neigh = [i+s for (i, s) in zip(
                                ind_shift, shift_new)]
                            #print(ind_shift, ind_shift_neigh)
                            ind_shift_neigh = np.asarray(ind_shift_neigh,
                                                         dtype=int)
                            if ind_shift_neigh[0]<0 or ind_shift_neigh[
                                0]==mask.shape[0]:
                                list_neighbours_valid.append(ind_shift)
                            elif ind_shift_neigh[1]<0 or ind_shift_neigh[
                                1]==mask.shape[1]:
                                list_neighbours_valid.append(ind_shift)
                            elif ind_shift_neigh[2]<0 or ind_shift_neigh[
                                2]==mask.shape[2]:
                                list_neighbours_valid.append(ind_shift)
                            elif status_array[ind_shift_neigh[0],
                                          ind_shift_neigh[1],
                                    ind_shift_neigh[2]]> -1 and status_array[
                                ind_shift_neigh[0], ind_shift_neigh[1],
                                    ind_shift_neigh[2]] < 2:
                                list_neighbours_valid.append(ind_shift_neigh)
                            else:
                                list_neighbours_valid.append(ind_shift)

                        f = 0
                        numel = 0
                        denom = 0
                        for (i, l) in enumerate(list_neighbours_valid):
                            if np.sum(np.asarray(l)-np.asarray(ind_shift)) != 0:
                                f += tangent_ind[i] * normed_grad[i] * (
                                    laplace_sol[ind_shift[0], ind_shift[1],
                                                ind_shift[2]] -
                                    laplace_sol[
                                    l[0], l[1], l[2]])/pixdim[i]
                                numel += np.abs(tangent_ind[i]) / pixdim[i] * \
                                         integral[l[0],l[1],l[2]]
                                denom += np.abs(tangent_ind[i]) / pixdim[i]
                        if denom <= 0:
                            for (i, l) in enumerate(list_neighbours_valid):
                                if np.sum(np.asarray(l) - np.asarray(
                                        ind_shift)) != 0:
                                    f += tangent_ind[i] * normed_grad[i] * (
                                            laplace_sol[
                                                ind_shift[0], ind_shift[1],
                                                ind_shift[2]] -
                                            laplace_sol[
                                                l[0], l[1], l[2]]) / pixdim[i]
                                    numel += 1.0 / pixdim[
                                        i] * \
                                             integral[l[0], l[1], l[2]]
                                    denom += 1.0 / pixdim[i]
                            integral[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = (to_integrate[ind_shift[0],
                                                    ind_shift[1], ind_shift[
                                                        2]] * denom + numel) / \
                                      denom
                            print('denom 0 dist is ',integral[ind_shift[0],
                                                              ind_shift[1], ind_shift[
                                2]] )

                            if integral[ind_shift[0], ind_shift[1], ind_shift[
                            2]] < 2000 and denom>0:
                                status_array[
                                    ind_shift[0], ind_shift[1], ind_shift[
                                        2]] = 1
                            else:
                                status_array[
                                    ind_shift[0], ind_shift[1], ind_shift[
                                        2]] = 1
                                integral[ind_shift[0], ind_shift[1], ind_shift[
                                    2]] = solve_average_integral(ind_shift,
                                                                status_array,\
                                         integral, to_integrate,pixdim)

                        else:
                            #print(distance.shape, ind_shift, numel, denom)
                            integral[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = (to_integrate[ind_shift[0],
                                       ind_shift[1], ind_shift[2]]+numel)/denom
                            print('denom normal dist is ', integral[
                                ind_shift[0], ind_shift[1], ind_shift[
                                2]])
                            status_array[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = 1
                        if status_array[ind_shift[0], ind_shift[1], ind_shift[
                                2]] == 1 and integral[ind_shift[0],
                                                      ind_shift[1],
                                                      ind_shift[2]] < 3000:
                            insert_ind = bisect_right(list_int, normalised_laplace[
                            ind_shift[0], ind_shift[1], ind_shift[
                                2]])
                            status_init = list_status[0:insert_ind,:]
                            status_end = list_status[insert_ind::, :]

                            if len(status_init) == 0:
                                list_status = np.vstack([ind_shift, list_status])
                            elif len(status_end) == 0:
                                list_status = np.vstack([list_status,ind_shift])
                            else:
                                list_status = np.vstack([status_init, ind_shift, \
                                          status_end])
                            list_int = list_int[0:insert_ind]+[normalised_laplace[ind_shift[
                                                                       0], ind_shift[1], ind_shift[
                                2]]] + list_int[insert_ind:]
                        elif status_array[ind_shift[0], ind_shift[1], ind_shift[
                                2]] == 1:
                            list_status = np.vstack([list_status, ind_shift])
                            list_int = list_int + [normalised_laplace[ind_shift[
                                                                       0], ind_shift[1], ind_shift[
                                2]]]

    return integral


def length_solving_laplace(distance, object, laplace_sol, pixdim, mask,
                           value=10000):

    border = object - binary_erosion(object)
    # normalise Laplace solution
    normalised_laplace = laplace_sol / value
    tangent = create_tangent_laplace(normalised_laplace, pixdim, mask)

    # Create status array
    status_array = np.ones_like(mask) * -1
    status_array = np.where(mask > 0, 2 * np.ones_like(mask), status_array)
    status_array = np.where(border, np.zeros_like(mask), status_array)
    status_array = np.where(distance>0, np.ones_like(mask), status_array)


    list_status_init = np.asarray(np.where(status_array == 1)).T
    list_dist = [distance[s[0],s[1],s[2]] for s in list_status_init]
    ind_sort = np.argsort(list_dist)
    list_status = list_status_init[ind_sort,:]
    list_dist = [list_dist[i] for i in ind_sort]
    while np.asarray(np.where(status_array==2)).shape[1] >0 and \
            list_status.shape[0] > 0:
        ind_interest = list_status[0, :]
        list_status = list_status[1:, :]
        list_dist = list_dist[1:]
        print(np.asarray(np.where(status_array==2)).shape, distance[ind_interest[0], ind_interest[1], ind_interest[2]])
        #print(ind_interest)
        status_array[ind_interest[0], ind_interest[1], ind_interest[2]] = 0
        #print(status_array[ind_interest[0], ind_interest[1], ind_interest[2]])
        for xi in range(-1, 2):
            for yi in range(-1, 2):
                for zi in range(-1, 2):
                    ind_shift = [xi+ind_interest[0], yi+ind_interest[1],
                                 zi+ind_interest[2]]
                    if abs(xi) + abs(yi) + abs(zi) == 1 and \
                            mask[ind_shift[0], ind_shift[1], ind_shift[2]] and \
                            status_array[ind_shift[0], ind_shift[1],
                                         ind_shift[2]] >1 and ind_shift[0]>0 \
                            and ind_shift[1] > 0 and ind_shift[2]>0 and \
                            ind_shift[0] < mask.shape[0] and ind_shift[1] < \
                            mask.shape[1] and ind_shift[2] < mask.shape[2]:


                        tangent_ind = tangent[ind_shift[0], ind_shift[1],
                                              ind_shift[2]]
                        normed_grad = np.sign(tangent_ind)

                        neigh_use = np.sign(tangent_ind)
                        print(tangent_ind, normed_grad, neigh_use)
                        list_neighbours_valid = []
                        for d in range(0, 3):
                            shift_new = [0, 0, 0]
                            shift_new[d] += neigh_use[d]
                            ind_shift_neigh = [i+s for (i, s) in zip(
                                ind_shift, shift_new)]
                            #print(ind_shift, ind_shift_neigh)
                            ind_shift_neigh = np.asarray(ind_shift_neigh,
                                                         dtype=int)
                            if ind_shift_neigh[0]<0 or ind_shift_neigh[
                                0]==mask.shape[0]:
                                list_neighbours_valid.append(ind_shift)
                            elif ind_shift_neigh[1]<0 or ind_shift_neigh[
                                1]==mask.shape[1]:
                                list_neighbours_valid.append(ind_shift)
                            elif ind_shift_neigh[2]<0 or ind_shift_neigh[
                                2]==mask.shape[2]:
                                list_neighbours_valid.append(ind_shift)
                            elif status_array[ind_shift_neigh[0],
                                          ind_shift_neigh[1],
                                    ind_shift_neigh[2]]> -1 and status_array[
                                ind_shift_neigh[0], ind_shift_neigh[1],
                                    ind_shift_neigh[2]] < 2:
                                list_neighbours_valid.append(ind_shift_neigh)
                            else:
                                list_neighbours_valid.append(ind_shift)

                        f = 0
                        numel = 0
                        denom = 0
                        for (i, l) in enumerate(list_neighbours_valid):
                            if np.sum(np.asarray(l)-np.asarray(ind_shift)) != 0:
                                f += tangent_ind[i] * normed_grad[i] * (
                                    laplace_sol[ind_shift[0], ind_shift[1],
                                                ind_shift[2]] -
                                    laplace_sol[
                                    l[0], l[1], l[2]])/pixdim[i]
                                numel += np.abs(tangent_ind[i]) / pixdim[i] * \
                                         distance[l[0],l[1],l[2]]
                                denom += np.abs(tangent_ind[i]) / pixdim[i]
                        if denom <= 0:
                            distance[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = \
                                solve_quadratic(
                                ind_shift,  status_array, distance, pixdim,
                                mask)
                            print('denom 0 dist is ',distance[ind_shift[0],
                                                              ind_shift[1], ind_shift[
                                2]] )
                            if distance[ind_shift[0], ind_shift[1], ind_shift[
                            2]] < 200:
                                status_array[
                                    ind_shift[0], ind_shift[1], ind_shift[
                                        2]] = 1
                        else:
                            #print(distance.shape, ind_shift, numel, denom)
                            distance[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = (1+numel)/denom
                            print('denom normal dist is ', distance[
                                ind_shift[0], ind_shift[1], ind_shift[
                                2]])
                            status_array[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = 1
                        insert_ind = bisect_right(list_dist, distance[ind_shift[0], ind_shift[1], ind_shift[
                                2]])
                        status_init = list_status[0:insert_ind,:]
                        status_end = list_status[insert_ind::, :]

                        if len(status_init) == 0:
                            list_status = np.vstack([ind_shift, list_status])
                        elif len(status_end) == 0:
                            list_status = np.vstack([list_status,ind_shift])
                        else:
                            list_status = np.vstack([status_init, ind_shift, \
                                          status_end])
                        list_dist = list_dist[0:insert_ind]+[distance[ind_shift[0], ind_shift[1], ind_shift[
                                2]]] + list_dist[insert_ind:]





def create_all_neighbours(index):
    list_neigh = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                if abs(x) + abs(y) + abs(z) > 0:
                    shift = [x, y, z]
                    index_shifted = index + shift
                    list_neigh.append(index_shifted)
    return list_neigh

def create_all_triplets_bis():
    list_triplets = [[
        [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [0, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [-1, 1, 1]],

        [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
         [[0, 1, 0], [0, 0, -1], [0, 1, -1]],
         [[0, 1, 0], [0, 1, -1], [1, 1, -1]],
         [[0, 1, 0], [0, 1, -1], [-1, 1, -1]],

        [[0, 1, 0], [0, 0, 1], [-1, 0, 0]],
         [[0, 1, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 1, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 1, 0], [1, 1, 0], [1, 1, -1]],

        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
         [[0, 1, 0], [-1, 0, 0], [-1, 1, 0]],
         [[0, 1, 0], [-1, 1, 0], [-1, 1, 1]],
         [[0, 1, 0], [-1, 1, 0], [-1, 1, -1]],

        [[0, -1, 0], [0, 0, 1], [1, 0, 0]],
         [[0, -1, 0], [0, 0, 1], [0, -1, 1]],
         [[0, -1, 0], [0, -1, 1], [1, -1, 1]],
         [[0, -1, 0], [0, -1, 1], [-1, -1, 1]],

        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
         [[0, -1, 0], [0, 0, -1], [0, -1, -1]],
         [[0, -1, 0], [0, -1, -1], [1, -1, -1]],
         [[0, -1, 0], [0, -1, -1], [-1, -1, -1]],

        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
         [[0, -1, 0], [1, 0, 0], [1, -1, 0]],
         [[0, -1, 0], [1, -1, 0], [1, -1, 1]],
         [[0, -1, 0], [1, -1, 0], [1, -1, -1]],

        [[0, -1, 0], [0, 0, -1], [-1, 0, 0]],
         [[0, -1, 0], [-1, 0, 0], [-1, 1, 0]],
         [[0, -1, 0], [-1, -1, 0], [-1, -1, 1]],
         [[0, -1, 0], [-1, -1, 0], [-1, -1, -1]]]
    return list_triplets, list_triplets[0:16]

def create_all_triplets():
    list_triplets =[[
    [0, 1, 0], [0, 0, 1], [1, 0, 0],
    [0, 1, 0], [0, 0, 1], [0, 1, 1],
    [0, 1, 0], [0, 1, 1], [1, 1, 1],
    [0, 1, 0], [0, 1, 1], [-1, 1, 1]],

    [[0, 1, 0], [0, 0, -1], [1, 0, 0],
    [0, 1, 0], [0, 0, -1], [0, 1, -1],
    [0, 1, 0], [0, 1, -1], [1, 1, -1],
    [0, 1, 0], [0, 1, -1], [-1, 1, -1]],

    [[0, 1, 0], [0, 0, 1], [-1, 0, 0],
    [0, 1, 0], [1, 0, 0], [1, 1, 0],
    [0, 1, 0], [1, 1, 0], [1, 1, 1],
    [0, 1, 0], [1, 1, 0], [1, 1, -1]],

    [[0, 1, 0], [0, 0, -1], [-1, 0, 0],
    [0, 1, 0], [-1, 0, 0], [-1, 1, 0],
    [0, 1, 0], [-1, 1, 0], [-1, 1, 1],
    [0, 1, 0], [-1, 1, 0], [-1, 1, -1]],

    [[0, -1, 0], [0, 0, 1], [1, 0, 0],
    [0, -1, 0], [0, 0, 1], [0, -1, 1],
    [0, -1, 0], [0, -1, 1], [1, -1, 1],
    [0, -1, 0], [0, -1, 1], [-1, -1, 1]],

    [[0, -1, 0], [0, 0, -1], [1, 0, 0],
    [0, -1, 0], [0, 0, -1], [0, -1, -1],
    [0, -1, 0], [0, -1, -1], [1, -1, -1],
    [0, -1, 0], [0, -1, -1], [-1, -1, -1]],

    [[0, -1, 0], [0, 0, 1], [-1, 0, 0],
    [0, -1, 0], [1, 0, 0], [1, -1, 0],
    [0, -1, 0], [1, -1, 0], [1, -1, 1],
    [0, -1, 0], [1, -1, 0], [1, -1, -1]],

    [[0, -1, 0], [0, 0, -1], [-1, 0, 0],
    [0, -1, 0], [-1, 0, 0], [-1, 1, 0],
    [0, -1, 0], [-1, -1, 0], [-1, -1, 1],
    [0, -1, 0], [-1, -1, 0], [-1, -1, -1]]]
    return list_triplets, list_triplets[0::4]


def create_tangent_laplace(laplace_to_tang, pixdim, mask):
    grad = np.gradient(laplace_to_tang, axis=(0, 1, 2))
    list_grad = []
    for (g, p) in zip(grad, pixdim):
        g = np.expand_dims(g * mask *p, -1)
        list_grad.append(g)
    full_grad = np.concatenate(list_grad, axis=-1)
    norm_gradient = np.sqrt(np.sum(np.square(full_grad), -1))
    tangent = full_grad / (np.expand_dims(norm_gradient, -1))
    tangent = np.nan_to_num(tangent)
    nib.save(nib.Nifti1Image(tangent, np.eye(4)),
             '/Users/csudre/Documents/Test500504P_D/TangentTest.nii.gz')
    return tangent


def solve_average_integral(index, status, integral, to_integrate, pixdim, \
                                                            value=10000):
    list_all, list_combi = create_all_triplets_bis()
    possible_dist = []
    for f in list_combi:
        list_known = []
        for (numb, shift) in enumerate(f):
            shifted_ind = [i + s for (i, s) in zip(index, shift)]
            if status[shifted_ind[0], shifted_ind[1], shifted_ind[2]] == 0:
                list_known.append(numb)
        if len(list_known) == 0:
            possible_dist.append(value)
        elif len(list_known) == 1:
            shift_known = f[list_known[0]]
            dir_shift = np.where(np.abs(shift_known) == 1)[0][0]
            pixdim_shift = pixdim[dir_shift]
            ind_shift = [i + s for (i, s) in zip(index, shift_known)]
            possible_dist.append(pixdim_shift * to_integrate[index[0],
                                                             index[1],
                                                             index[2]] +
                                 integral[ind_shift[0],ind_shift[1],ind_shift[2]])
        else:
            sum_int = 0
            sum_den = 0
            sum_dist = 0
            for k in range(0, len(list_known)):
                shift_known = f[list_known[k]]
                dir_shift = np.where(np.abs(shift_known) == 1)[0][0]
                sum_den += 1.0 / pixdim[dir_shift]
                sum_dist += np.square(pixdim[dir_shift])
                ind_shift = [i + s for (i, s) in zip(index, shift_known)]

                sum_int += integral[ind_shift[0], ind_shift[1], ind_shift[
                    2]] / pixdim[dir_shift]
            possible_dist.append(to_integrate[index[0], index[1],
                                              index[2]]+sum_int/sum_den)
    return np.min(possible_dist)


def solve_quadratic(index, status, distance, pixdim, mask, value=10000):
    list_all, list_combi = create_all_triplets_bis()
    possible_dist = []
    for f in list_combi:
        list_known = []
        for (numb, shift) in enumerate(f):
            shifted_ind = [ i+s for (i,s) in zip(index, shift)]
            if status[shifted_ind[0], shifted_ind[1], shifted_ind[2]] == 0:
                list_known.append(numb)
        if len(list_known) == 0:
            possible_dist.append(value)
        elif len(list_known) == 1:
            shift_known = f[list_known[0]]
            dir_shift = np.where(np.abs(shift_known)==1)[0][0]
            pixdim_shift = pixdim[dir_shift]
            ind_shift = [i+s for (i,s) in zip(index, shift_known)]
            possible_dist.append(pixdim_shift + distance[ind_shift[0],
                                                         ind_shift[1],
                                                         ind_shift[2]])
        else:
            a = 0
            b = 0
            c = 0
            for k in range(0, len(list_known)):
                shift_known = f[list_known[k]]
                dir_shift = np.where(np.abs(shift_known)==1)[0][0]
                a+= np.square(1.0 / pixdim[dir_shift])
                ind_shift = [i+s for (i,s) in zip(index, shift_known)]

                b -= 2 * distance[ind_shift[0], ind_shift[1], ind_shift[2]] / \
                     np.square(pixdim[dir_shift])
                c += np.square(distance[ind_shift[0], ind_shift[1],
                                        ind_shift[2]] / pixdim[dir_shift])
            d = np.square(b) - 4 * a * (c - 1)
            if d > 0:
                possible_dist.append(-b+np.sqrt(d) / (2 * a))
            else:
                minTest = 1E18
                for k in range(0, len(list_known)):
                    shift_known = f[list_known[k]]
                    dir_shift = np.where(np.abs(shift_known) == 1)[0][0]
                    ind_shift = [i+s for (i, s) in zip(index, shift_known)]
                    if pixdim[dir_shift]+distance[ind_shift[0], ind_shift[1],
                                                  ind_shift[2]] < minTest:
                        minTest = pixdim[dir_shift]+distance[ind_shift[0],\
                            ind_shift[1], ind_shift[2]]
                possible_dist.append(minTest)
    return np.min(possible_dist)

def create_normalised_dist(object_in, object_out, mask, pixdim, value=10000):
    surface_in = object_in - binary_erosion(object_in)
    surface_out = object_out - binary_erosion(object_out)
    laplace_sol = solving_laplace_equation(mask, surface_out, surface_in,
                                          pixdim, value, thresh_conv=0.00001)
    norm_length = normalised_laplace_length(laplace_sol, object_in, object_out,
                                     pixdim, mask)
    return laplace_sol, norm_length




path = '/Users/csudre/Documents/UK_BBK/TempWork/1394248'
connect_nii = nib.load(os.path.join(path, 'Connect_1394248.nii.gz'))
connect_data = connect_nii.get_data()
mask_lesion = np.where(connect_data==8, np.ones_like(connect_data),
                       np.zeros_like(connect_data))
flair_nii = nib.load(os.path.join(path, 'FLAIR_1394248.nii.gz'))
flair_data = flair_nii.get_data()

skeleton, laplace_sol, integral = lesion_laplace_integral(flair_data,
                                                      mask_lesion,[1,1,1])


new_lapnii = nib.Nifti1Image(laplace_sol, flair_nii.affine)
new_intnii = nib.Nifti1Image(integral, flair_nii.affine)
new_sknii = nib.Nifti1Image(skeleton, flair_nii.affine)
nib.save(new_sknii, os.path.join(path, 'Skel8_1394248.nii.gz'))
nib.save(new_lapnii, os.path.join(path, 'LapSol8_1394248.nii.gz'))
nib.save(new_intnii, os.path.join(path, 'Int8_1394248.nii.gz'))
print('Lap finished')

path = '/Users/csudre/Documents/GIFOut_T1_500504P_down'
parc_file = os.path.join(path,'T1_500504P_down_NeuroMorph_Parcellation.nii.gz')
parc_nii = nib.load(parc_file)
parc = parc_nii.get_data()
ventr = np.where(np.logical_and(parc>49.5, parc<53.5), np.ones_like(parc),
                 np.zeros_like(parc))
brain = np.where(parc>4.5, np.ones_like(parc), np.zeros_like(parc))
mask = np.copy(brain)
laplace_sol, norm_dist = create_normalised_dist(ventr, brain, brain,
                                                parc_nii.header.get_zooms())
new_lapnii = nib.Nifti1Image(laplace_sol, parc_nii.affine)
new_normnii = nib.Nifti1Image(norm_dist, parc_nii.affine)
nib.save(new_lapnii, os.path.join(path, 'LapSol_500504P_down.nii.gz'))
nib.save(new_normnii, os.path.join(path, 'NormDist_500504P_down.nii.gz'))
print('Lap finished')




















