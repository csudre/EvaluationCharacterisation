
import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.morphology import distance_transform_edt as dist_edt
from bisect import bisect_right
import trimesh
from skimage import measure
from skimage.morphology import skeletonize_3d
from skimage import measure
from scipy.spatial.distance import cdist
import os
import nibabel as nib
import argparse
import sys
from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay as DL

from scipy.ndimage import label, gaussian_filter1d, gaussian_gradient_magnitude


def create_hills(distance):
    """
    Create hill map based on distance map
    :param distance:
    :return: hill map
    """

    results = np.zeros_like(distance)
    spatial_rank = 3
    #  Creation in each spatial direction of the rolled gradients of the
    # distance map for calculation of second derivative.
    for i in range(0, spatial_rank):
        pad_array = [[0, 0], [0, 0], [0, 0]]
        pad_array2 = [[0, 0], [0, 0], [0, 0]]
        pad_array[i] = [1, 0]
        pad_array2[i] = [0, 1]
        dist_plus = np.roll(distance, shift=1, axis=i)
        dist_minus = np.roll(distance, shift=-1, axis=i)
        test_diff = dist_minus - distance
        test_diff2 = distance - dist_plus

        sign_diff = np.sign(test_diff) + np.sign(test_diff2)
        grad_sign = np.where(sign_diff== 0, np.ones_like(
            distance),np.zeros_like(distance)) * np.where(
            test_diff<= 0, np.ones_like(distance),
            np.zeros_like(distance))
        results += grad_sign
    return results

def curvature_trimesh(binary, pixdim):
    verts, faces, normals, values = measure.marching_cubes_lewiner(binary, 0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    border = binary - binary_erosion(binary, iterations=1)
    indices_border = list(np.asarray(np.where(border)).T)
    curvature_res = trimesh.curvature.discrete_mean_curvature_measure(mesh,
                                                         indices_border, 1)

    curv_res = np.zeros_like(binary).astype(float)
    for (f, c) in zip(indices_border, curvature_res):
        curv_res[f[0], f[1], f[2]] = c
    return curv_res

def curvature(binary, pixdim):
    edges = []
    norm_edges = []
    verts, faces, normals, values = measure.marching_cubes_lewiner(binary, 0)
    curv_vect = [[] for v in verts]
    for f in faces:
        if (verts[f[0]]==verts[f[1]]).all() or (verts[f[1]]==verts[f[
            2]]).all() or (verts[f[0]]==verts[f[2]]).all():
            #print("degenerated face", verts[f[0]], verts[f[1]], verts[f[2]])
            continue
        e0 = np.sort([f[0], f[1]])
        e1 = np.sort([f[0], f[2]])
        e2 = np.sort([f[1], f[2]])
        if list(e0) not in edges:
            edges.append(list(e0))
            curv_t = curv_temp(verts[e0[0]],verts[e0[1]],normals[e0[
                0]],normals[e0[1]],pixdim)
            norm_edges.append(curv_t)
            curv_vect[f[0]].append(curv_t)
            curv_vect[f[1]].append(curv_t)
        if list(e1) not in edges:
            edges.append(list(e1))
            norm_edges.append(curv_temp(verts[e1[0]],verts[e1[1]],normals[e1[
                0]],normals[e1[1]],pixdim ))
            curv_t = curv_temp(verts[e1[0]], verts[e1[1]], normals[e1[
                0]], normals[e1[1]], pixdim)
            norm_edges.append(curv_t)
            curv_vect[f[0]].append(curv_t)
            curv_vect[f[2]].append(curv_t)
        if list(e2) not in edges:
            edges.append(list(e2))
            norm_edges.append(curv_temp(verts[e2[0]],verts[e2[1]],normals[e2[
                0]],normals[e2[1]], pixdim ))
            curv_t = curv_temp(verts[e2[0]], verts[e2[1]], normals[e2[
                0]], normals[e2[1]], pixdim)
            norm_edges.append(curv_t)
            curv_vect[f[1]].append(curv_t)
            curv_vect[f[2]].append(curv_t)
    for v in range(0, len(verts)):
        # edgcurv= [(e, c) for (e,c) in zip(edges,norm_edges) if v in e]
        # curv_v = [ec[1] for ec in edgcurv]
        curv_v = curv_vect[v]
        curvature_finv = np.sign(np.prod(curv_v))*np.power(np.abs(np.prod(
            curv_v)), 1.0/3)
        values[v] = curvature_finv
    return verts, normals, edges, faces, values

def assignment_curvature(binary, verts, values):
    border = binary - binary_erosion(binary, iterations=1)
    indices = list(np.asarray(np.where(border)).T)
    assigned = np.zeros_like(binary).astype(float)
    missing_indices = []
    for ind in indices:
        verts_diff = verts - ind
        sum_absdiff = np.sum(np.abs(verts_diff),1)
        min_absdiff = np.min(sum_absdiff)
        ind_min = [ i for i in range(0, len(sum_absdiff)) if sum_absdiff[i]
                    == min_absdiff]
        val_toave = [values[i] for i in ind_min]
        assigned[ind[0], ind[1], ind[2]] = np.mean(val_toave)
    return assigned



def curv_temp(vert0, vert1, norm0, norm1, pixdim):
    dist = np.sum(np.square((vert0-vert1)*pixdim))
    dot = np.dot((norm1-norm0)*pixdim, (vert1-vert0)*pixdim)
    return dot/dist

def curvature_image(binary):
    #first and second derivative
    dalpha = np.pi/1000
    x1 = gaussian_filter1d(binary, sigma=1, order=1, axis=0, mode='wrap')
    x2 = gaussian_filter1d(x1, sigma=1, order=1, axis=0, mode='wrap')

    y1 = gaussian_filter1d(binary, sigma=1, order=1, axis=1, mode='wrap')
    y2 = gaussian_filter1d(y1, sigma=1, order=1, axis=1, mode='wrap')

    z1 = gaussian_filter1d(binary, sigma=1, order=1, axis=2, mode='wrap')
    z2 = gaussian_filter1d(z1, sigma=1, order=1, axis=2, mode='wrap')

    return np.sqrt(np.square(x1*y2 - y1*x2) + np.square(x1*z2 - z1 * x2) +
                   np.square(y1 * z2 - y2 *z1)) / \
           gaussian_gradient_magnitude(binary, sigma=1)



def lesion_laplace_integral(intensities, lesion_mask, pixdim, value=10000,
                          thresh_conv=0.00005):
    #dil_mask = binary_dilation(lesion_mask, iterations=3).astype(int)
    dil_mask = lesion_mask
    skeleton = skeletonize_3d(dil_mask).astype(int)
    border_lesion = dil_mask - binary_erosion(dil_mask, iterations=1)
    # skeleton -= border_lesion.astype(int) * skeleton
    print(np.sum(skeleton), np.sum(lesion_mask), np.sum(border_lesion))
    laplace_sol = solving_laplace_equation(dil_mask, border_lesion,
                                           skeleton, pixdim, value,
                                           thresh_conv)*dil_mask
    print(np.unique(laplace_sol))
    integral = integral_along_laplace(intensities, skeleton, laplace_sol,
                                   pixdim,
                           dil_mask, value)
    dip = create_dip_border(border_lesion, integral, pixdim)
    print(np.max(dip), np.count_nonzero(dip), np.count_nonzero(border_lesion))
    return skeleton, laplace_sol, integral, dip


def solving_laplace_equation(mask, surface_out, surface_in, pixdim, value=10000,
                             thresh_conv=0.00005):
    old_flux = initialise_laplace(mask, surface_out, surface_in,
                                  value).astype(float)

    old_energy = 0.00001
    flag_continue = True
    while(flag_continue):
        current_flux = update_flux(old_flux, mask, pixdim)
        current_flux = refine_flux(current_flux, surface_in, surface_out)
        new_energy = total_energy_calculation(current_flux, surface_out,
                                              surface_in, mask, pixdim)
        # print(new_energy, np.abs(new_energy-old_energy)/old_energy)
        if(np.abs(new_energy-old_energy)/old_energy<=thresh_conv) or \
                new_energy==0:
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
                    shift_final += shift_temp * 1.0/normalisation
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
    #print(np.max(dist_in))
    length_solving_laplace(dist_out, object_out, laplace_sol*-1, pixdim, mask)
    #print(np.max(dist_out))
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
        # print(np.asarray(np.where(status_array==2)).shape, integral[
        #     ind_interest[0], ind_interest[1], ind_interest[2]])
        # #print(ind_interest)
        # print(normalised_laplace[ind_interest[0],ind_interest[1],
        #                          ind_interest[2]])
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
                        # print(tangent_ind, normed_grad, neigh_use)
                        list_neighbours_valid = []
                        for d in range(0, 3):
                            shift_new = [0, 0, 0]
                            shift_new[d] += neigh_use[d]
                            ind_shift_neigh = [i+s for (i, s) in zip(
                                ind_shift, shift_new)]
                            # print(ind_shift, ind_shift_neigh)
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
                            # print('denom 0 dist is ',integral[ind_shift[0],
                            #                                   ind_shift[1], ind_shift[
                            #     2]] )

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
                            # print('denom normal dist is ', integral[
                            #     ind_shift[0], ind_shift[1], ind_shift[
                            #     2]])
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
        # print(np.asarray(np.where(status_array==2)).shape, distance[ind_interest[0], ind_interest[1], ind_interest[2]])
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
                        # print(tangent_ind, normed_grad, neigh_use)
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
                            # print('denom 0 dist is ',distance[ind_shift[0],
                            #                                   ind_shift[1], ind_shift[
                            #     2]] )
                            if distance[ind_shift[0], ind_shift[1], ind_shift[
                            2]] < 200:
                                status_array[
                                    ind_shift[0], ind_shift[1], ind_shift[
                                        2]] = 1
                        else:
                            #print(distance.shape, ind_shift, numel, denom)
                            distance[ind_shift[0], ind_shift[1], ind_shift[
                                2]] = (1+numel)/denom
                            # print('denom normal dist is ', distance[
                            #     ind_shift[0], ind_shift[1], ind_shift[
                            #     2]])
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

def convexhull_fromles(lesion):
        border_periv = lesion - binary_erosion(lesion)
        indices = np.asarray(np.where(border_periv)).T
        out_idx = indices
        if indices.shape[0] > 4:
            try:
                test_chi = CH(indices)
                deln = DL(test_chi.points[test_chi.vertices])
                idx = np.stack(np.indices(lesion.shape), axis=-1)
                out_idx = np.asarray(np.nonzero(deln.find_simplex(idx) + 1)).T
            except:
                out_idx = indices
        else:
            out_idx = indices
        max_shape = np.tile(np.expand_dims(np.asarray(border_periv.shape)-1,
                                           0), [np.asarray(out_idx).shape[0],1])
        out_idx = np.minimum(np.asarray(out_idx), max_shape)
        print("Max of ch indices", np.max(out_idx, 0))
        chi_temp = np.zeros(lesion.shape)
        chi_temp[list(out_idx.T)] = 1
        return chi_temp

def comparison_convex_hull(lesion_mask, pixdim):
    convex_hull_map = convexhull_fromles(lesion_mask)
    border_ch = convex_hull_map - binary_erosion(convex_hull_map)
    border_les = lesion_mask - binary_erosion(lesion_mask)
    dist_ch = dist_edt(convex_hull_map, sampling=pixdim)
    skeleton = skeletonize_3d(lesion_mask)
    dist_skel = dist_edt(binary_dilation(convex_hull_map,
                                         iterations=4)-skeleton,
                         sampling=pixdim)
    dist_chfin = (dist_ch + 1.0) * border_les
    dist_skelfin = (dist_skel + 1.0) * border_les

    return dist_chfin, np.nan_to_num(dist_chfin/dist_skelfin)


def comparison_convex_hull_lap(lesion_mask, pixdim):
    convex_hull_map = convexhull_fromles(lesion_mask)
    border_ch = convex_hull_map - binary_erosion(convex_hull_map)
    border_les = lesion_mask - binary_erosion(lesion_mask)
    skeleton = skeletonize_3d(lesion_mask)
    dist_inles = initial_distance(skeleton, pixdim, lesion_mask, 1)
    laplace_sol_les = solving_laplace_equation(lesion_mask, border_les,
                                               skeleton,
                                           pixdim, thresh_conv=0.00001)
    laplace_sol_ch = solving_laplace_equation(convex_hull_map, border_ch,
                                               skeleton,
                                           pixdim, thresh_conv=0.00001)
    length_solving_laplace(dist_inles, skeleton, laplace_sol_les, pixdim,
                           lesion_mask)
    dist_inch = initial_distance(border_ch, pixdim, convex_hull_map, 1)
    length_solving_laplace(dist_inch, border_ch, -1.0 * laplace_sol_ch, pixdim,
                           convex_hull_map)
    dist_inles = initial_distance(skeleton, pixdim, lesion_mask, 1)
    length_solving_laplace(dist_inles, skeleton, laplace_sol_les, pixdim,
                           lesion_mask)
    ratio_dist_border = dist_inch / dist_inles * border_les
    dist_ch = dist_inch * border_les
    return dist_ch, ratio_dist_border




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

def create_weight_border(index, border, pixdim):
    X,Y,Z = np.meshgrid([1,0,1],[1,0,1],[1,0,1])
    dist = np.sqrt(np.square(pixdim[0])*X+np.square(pixdim[1])*Y+np.square(
        pixdim[2])*Z)
    dist = np.reciprocal(dist)
    dist[1,1,1] = 0
    x = index[0]
    y = index[1]
    z = index[2]
    if not border[x, y, z]:
        return 0
    border_temp = border[index[0]-1:index[0]+2, index[1]-1:index[1]+2,
                  index[2]-1: index[2]+2]
    weight_border = np.sum(border_temp * dist)/np.sum(dist)
    #print(weight_border)
    return weight_border

def create_dip_border(border, integral, pixdim):
    integral_fromborder = integral * border
    weight_init = np.zeros_like(border).astype(float)
    weight = np.copy(weight_init)
    dip = np.zeros_like(border)
    indices_border = np.asarray(np.where(border)).T
    for ind in indices_border:
        x=ind[0]
        y=ind[1]
        z=ind[2]
        val_weight = create_weight_border(ind, border, pixdim)
        weight[x, y, z] += val_weight
        #print(weight[x,y,z])
    print(np.count_nonzero(weight))
    for ind in indices_border:
        x = ind[0]
        y = ind[1]
        z = ind[2]
        values = np.ones([3, 3]) * integral[x,y,z]
        diff = integral[ind[0]-1:ind[0]+2, ind[1]-1:ind[1]+2, ind[2]-1:ind[
                                                                           2]+2] - values
        diff_pos = np.where(diff>0, diff, np.zeros_like(diff))

        weights_temp = weight[ind[0]-1:ind[0]+2, ind[1]-1:ind[1]+2, ind[2]-1:ind[
                                                                           2]+2]
        # print("weights sum and diff", np.sum(diff_pos), np.sum(weights_temp))
        dip[x, y, z] = np.sum(diff_pos * weights_temp)

    return dip

def main(argv):

    parser = argparse.ArgumentParser(description='Create Laplace solution for each relevant lesion')
    parser.add_argument('-les', dest='lesion', metavar='input pattern',
                        type=str,
                        help='RegExp pattern for the lesion files')
    parser.add_argument('-mahal', dest='mahal', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mahal files')
    parser.add_argument('-name', dest='name', action='store',
                        default='', type=str,
                        help='RegExp pattern for the name ')
    parser.add_argument('-parc', dest='parc', action='store', default='',
                        type=str)
    parser.add_argument('-type', dest='type', default='cc', choices=['cc','ventr'])
    parser.add_argument('-dist', dest='dist', default='in', choices=['norm', ' \
                                                                        ''in',
                                                            'out'])

    try:
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compute_ROI_statistics.py -les <input_image_pattern> -mahal '
              '<mask_image_pattern> -type -dist -parc -name   ')
        sys.exit(2)

    if args.type == 'cc':
        lesion_nii = nib.load(args.lesion)
        pixdim = lesion_nii.header.get_zooms()
        path = os.path.split(args.lesion)[0]
        mask = lesion_nii.get_data()
        binary_mask = np.where(mask>0.5, np.ones_like(mask), np.zeros_like(
            mask) )
        mahal_data1 = None
        mahal_data2 = None
        if args.mahal != '' and os.path.exists(args.mahal):
            mahal_nii = nib.load(args.mahal)
            mahal_data = mahal_nii.get_data()
            if mahal_data.ndim == 4:
                mahal_data1 = mahal_data[..., -1]
                mahal_data2 = -1 * mahal_data[..., -2]
            else:
                mahal_data1 = mahal_data
        else:
            mahal_data1 = mask.astype(int)

        if mahal_data2 is not None:
            hills2 = create_hills(mahal_data2) * binary_mask
            hills_nii = nib.Nifti1Image(hills2, lesion_nii.affine)
            nib.save(hills_nii, os.path.join(
                path, 'HillsT1_' + args.name + '.nii.gz'))
        hills1 = create_hills(mahal_data1) * binary_mask
        hills_nii = nib.Nifti1Image(hills1, lesion_nii.affine)
        nib.save(hills_nii, os.path.join(
            path, 'HillsFLAIR_' + args.name + '.nii.gz'))
        lab_map, numb_lab = label(mask)
        border = binary_mask - binary_erosion(binary_mask)
        print("Beginning curving")
        curving = curvature_trimesh(binary_mask, pixdim)
        print("Finished curving")
        curv_nii1 = nib.Nifti1Image(curving, lesion_nii.affine)
        nib.save(curv_nii1, os.path.join(
            path, 'CurvTM_' + args.name + '.nii.gz'))
        #verts, normals, edges, faces, values = curvature(binary_mask, pixdim)

        # assigned_curve = assignment_curvature(binary_mask, verts,values)
        # curv_nii = nib.Nifti1Image(assigned_curve, lesion_nii.affine)
        # nib.save(curv_nii, os.path.join(
        #     path, 'Curv_'+args.name+'.nii.gz'))




        cc_map = measure.label(mask, connectivity=3,
                              background=0).astype(float)

        cc_nii = nib.Nifti1Image(lab_map, lesion_nii.affine)
        nib.save(cc_nii, os.path.join(
            path, 'CC_'+args.name+'.nii.gz'))
        skeleton_fin = np.zeros_like(cc_map).astype(int)
        integral_fin = np.zeros_like(cc_map).astype(float)
        distance_fin = np.zeros_like(cc_map).astype(float)
        laplace_fin = np.zeros_like(cc_map).astype(float)
        dip_fin = np.zeros_like(cc_map).astype(float)
        dist_chfin = np.zeros_like(cc_map).astype(float)
        ratio_chfin = np.zeros_like(cc_map).astype(float)
        values_lab = np.unique(lab_map)[1: ]
        for val in values_lab:
            indices_lab = np.asarray(np.where(lab_map == val)).T


            if indices_lab.shape[0] > 30:
                print("Creation of laplacian for lesion ", val, 'with volume '
                                                                '',
                      indices_lab.shape[0])

                temp_seg = np.where(lab_map == val, np.ones_like(lab_map),
                                    np.zeros_like(lab_map))
                skeleton, laplace_sol, integral, dip = lesion_laplace_integral(
                    mahal_data1, temp_seg, pixdim)
                dist_ch, ratio_distch = comparison_convex_hull(temp_seg,
                                                               pixdim)
                lap2, distance = create_normalised_dist(skeleton, temp_seg,
                                                  temp_seg, pixdim)
                skeleton_fin += skeleton
                distance_fin += distance
                integral_fin += integral
                laplace_fin += laplace_sol
                dip_fin += dip
                dist_chfin += dist_ch
                ratio_chfin += ratio_distch
        data_fin = np.concatenate([
            np.expand_dims(skeleton_fin, -1),
            np.expand_dims(laplace_fin, -1),
            np.expand_dims(distance_fin, -1),
            np.expand_dims(integral_fin, -1),
            np.expand_dims(dip_fin, -1),
            np.expand_dims(dist_chfin, -1),
            np.expand_dims(ratio_chfin, -1)], -1)

        new_sknii = nib.Nifti1Image(data_fin, lesion_nii.affine)

        nib.save(new_sknii, os.path.join(path, 'PLLap_'+args.name+'.nii.gz'))

    elif args.type == 'ventr':
        path = os.path.split(args.parc)[0]
        parc_nii = nib.load(args.parc)
        parc = parc_nii.get_data()
        ventr = np.where(np.logical_and(parc > 49.5, parc < 53.5),
                         np.ones_like(parc),
                         np.zeros_like(parc))
        brain = np.where(parc > 4.5, np.ones_like(parc), np.zeros_like(parc))
        mask = np.copy(brain)
        laplace_sol, norm_dist = create_normalised_dist(ventr, brain, brain,
                                                        parc_nii.header.get_zooms())
        data_fin = np.concatenate([
            np.expand_dims(laplace_sol, -1),
            np.expand_dims(norm_dist, -1)], -1)
        new_lapnii = nib.Nifti1Image(data_fin, parc_nii.affine)
        nib.save(new_lapnii, os.path.join(path,
                                          'LapSolNorm_'+args.name+
                                          'nii.gz'))


# path = '/Users/csudre/Documents/UK_BBK/TempWork/1394248'
# connect_nii = nib.load(os.path.join(path, 'Connect_1394248.nii.gz'))
# connect_data = connect_nii.get_data()
# mask_lesion = np.where(connect_data==8, np.ones_like(connect_data),
#                        np.zeros_like(connect_data))
# flair_nii = nib.load(os.path.join(path, 'FLAIR_1394248.nii.gz'))
# flair_data = flair_nii.get_data()
#
# skeleton, laplace_sol, integral = lesion_laplace_integral(flair_data,
#                                                       mask_lesion,[1,1,1])
#
#
# new_lapnii = nib.Nifti1Image(laplace_sol, flair_nii.affine)
# new_intnii = nib.Nifti1Image(integral, flair_nii.affine)
# new_sknii = nib.Nifti1Image(skeleton, flair_nii.affine)
# nib.save(new_sknii, os.path.join(path, 'Skel8_1394248.nii.gz'))
# nib.save(new_lapnii, os.path.join(path, 'LapSol8_1394248.nii.gz'))
# nib.save(new_intnii, os.path.join(path, 'Int8_1394248.nii.gz'))
# print('Lap finished')
#
# path = '/Users/csudre/Documents/GIFOut_T1_500504P_down'
# parc_file = os.path.join(path,'T1_500504P_down_NeuroMorph_Parcellation.nii.gz')
# parc_nii = nib.load(parc_file)
# parc = parc_nii.get_data()
# ventr = np.where(np.logical_and(parc>49.5, parc<53.5), np.ones_like(parc),
#                  np.zeros_like(parc))
# brain = np.where(parc>4.5, np.ones_like(parc), np.zeros_like(parc))
# mask = np.copy(brain)
# laplace_sol, norm_dist = create_normalised_dist(ventr, brain, brain,
#                                                 parc_nii.header.get_zooms())
# new_lapnii = nib.Nifti1Image(laplace_sol, parc_nii.affine)
# new_normnii = nib.Nifti1Image(norm_dist, parc_nii.affine)
# nib.save(new_lapnii, os.path.join(path, 'LapSol_500504P_down.nii.gz'))
# nib.save(new_normnii, os.path.join(path, 'NormDist_500504P_down.nii.gz'))
# print('Lap finished')
#
#

if __name__ == "__main__":
    main(sys.argv[1:])

















