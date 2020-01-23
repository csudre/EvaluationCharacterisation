from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.ma as ma
import math
import pylab as pl
import scipy.stats.mstats as mstats
import scipy.ndimage as ndimage
from skimage.morphology import convex_hull_image as chi
from scipy.spatial import ConvexHull as CH
from scipy.spatial import Delaunay as DL
import scipy.ndimage.morphology as morph
from .morphology import MorphologyOps
from .pairwise_measures import CacheFunctionOutput
# from niftynet.utilities.misc_common import MorphologyOps, CacheFunctionOutput
LIST_HARALICK = ['asm', 'contrast', 'correlation', 'sumsquare',
                 'sum_average', 'idifferentmomment', 'sumentropy', 'entropy',
                 'differencevariance', 'sumvariance', 'differenceentropy',
                 'imc1', 'imc2']
LIST_SHAPE = ['centre of mass', 'volume', 'surface', 'ratio_eigen', 'fa',
              'solidity', 'compactness', 'contour_smoothness', 'circularity',
              'balance', 'eigen_values', 'max_dist_com', 'fractal_dim']
LIST_HIST = ['weighted_mean', 'weighted_std', 'median', 'skewness',
             'kurtosis', 'wquantile_1', 'wquantile_5', 'wquantile_25',
             'wquantile_50', 'wquantile_75', 'wquantile_95', 'wquantile_99']


class RegionProperties(object):
    def __init__(self, seg, img, measures,
                 num_neighbors=6, threshold=0, pixdim=None, bin_numb=100,
                 bin_glszm=32, mul=10, trans=50, lab_channels=None):
        if pixdim is None:
            pixdim = [1, 1, 1]
        self.seg = seg
        self.bin_seg = np.where(self.seg>0, np.ones_like(self.seg),
                                np.zeros_like(self.seg))
        self.bin = bin_numb
        self.bin_glszm = bin_glszm
        self.mul = mul  # 10 for Norm 5 for Z 50 for Stacked
        self.trans = trans  # 50 for all
        if img.ndim == 3:
            img = np.expand_dims(np.expand_dims(img, -1), -1)
        if img.ndim == 4:
            img = np.expand_dims(img, 3)
        self.img = img
        self.measures = measures
        self.m_dict_result = {}

        self.pixdim = pixdim
        self.threshold = threshold
        if any(LIST_HARALICK) in measures or 'asm' in measures:
            self.haralick_flag = True
        else:
            self.haralick_flag = False
        #print("Harilick flag", self.haralick_flag)

        self.vol_vox = np.prod(pixdim)
        self.img_channels = self.img.shape[4] if img.ndim == 5 else 1
        img_id_init = range(0, self.img_channels)
        img_id = []
        if lab_channels is not None:
            img_id = [l for (i, l) in enumerate(lab_channels) if i <
                      self.img_channels]
            if len(lab_channels) < self.img_channels:
                for id in range(0, self.img_channels - len(lab_channels)):
                    img_id.append(str(len(lab_channels)+id))
        else:
            for l in img_id_init:
                img_id.append(str(l))
        self.img_id = img_id
        if self.seg is not None:
            self.masked_img, self.masked_seg = self.__compute_mask()
        else:
            print("no mask")
        self.neigh = num_neighbors
        # self.harilick_m = np.atleast_2d(self.harilick_matrix())
        if self.haralick_flag:
            self.harilick_m = np.atleast_2d(self.harilick_matrix())
        # self.glszm = self.grey_level_size_matrix()
        self.m_dict = {
            'centre of mass': (self.centre_of_mass, ['CoMx',
                                                     'CoMy',
                                                     'CoMz']),
            'volume': (self.volume,
                       ['NVoxels', 'NVoxelsBinary', 'Vol', 'VolBinary']),
            'max_dist_com': (self.max_dist_com, ['MaxDistCoM']),
            'surface': (self.surface, ['NSurface',
                                       'NSurfaceBinary',
                                       'SurfaceVol',
                                       'SurfaceVolBinary']),
            'surface volume ratio': (self.sav, ['SAVNumb',
                                                'SAVNumBinary',
                                                'SAV',
                                                'SAVBinary']),
            'compactness': (self.compactness, ['CompactNumb',
                                               'CompactNumbBinary',
                                               'Compactness',
                                               'CompactnessBinary']),
            'solidity': (self.solidity, ['Solidity']),
            'balance': (self.balanced_rep, ['Balance']),
            'fractal_dim': (self.fractal_dimension, ['Fractal_dim']),
            'circularity': (self.circularity, ['Circularity']),
            'contour_smoothness': (self.contour_smoothness, ['Contour_Smooth']),
            'eigen_values': (self.ellipsoid_lambdas, ['lambda_%i' % i for i
                                                      in range(0,3)]),
            'ratio_eigen': (self.ratio_eigen, ['ratio_eigen_%i' % i for i in
                                               range(0,3)]),
            'fa': (self.fractional_anisotropy, ['FA']),
            'mean': (self.mean_, ['Mean_%s' % i for i in img_id]),
            'weighted_mean': (self.weighted_mean_,
                              ['Weighted_mean_%s' % i for i in img_id]),
            'weighted_std': (self.weighted_std_, ['Weighted_std_%s' % i for i
                                                  in img_id]),
            'median': (self.median_, ['Median_%s' % i for i in img_id]),
            'skewness': (self.skewness_, ['Skewness_%s' % i for i in img_id]),
            'kurtosis': (self.kurtosis_, ['Kurtosis_%s' % i for i in img_id]),
            'min': (self.min_ if np.sum(self.seg) > 0 else
                    self.return_0, ['Min_%s' % i for i in img_id]),
            'max': (self.max_ if np.sum(self.seg) > 0 else
                    self.return_0, ['Max_%s' % i for i in img_id]),
            'quantile_1': (self.quantile_1 if np.sum(self.seg) > 0 else
                           self.return_0, ['P1_%s' % i for i in img_id]),
            'quantile_5': (self.quantile_5 if np.sum(self.seg) > 0 else
                           self.return_0, ['P5_%s' % i for i in img_id]),
            'quantile_25': (self.quantile_25 if np.sum(self.seg) > 0 else
                            self.return_0, ['P25_%s' % i for i in img_id]),
            'quantile_50': (self.median_ if np.sum(self.seg) > 0 else
                            self.return_0, ['P50_%s' % i for i in img_id]),
            'quantile_75': (self.quantile_75 if np.sum(self.seg) > 0 else
                            self.return_0, ['P75_%s' % i for i in img_id]),
            'quantile_95': (self.quantile_95 if np.sum(self.seg) > 0 else
                            self.return_0, ['P95_%s' % i for i in img_id]),
            'quantile_99': (self.quantile_99 if np.sum(self.seg) > 0 else
                            self.return_0, ['P99_%s' % i for i in img_id]),

            'wquantile_1': (self.weighted_quantile_1 if np.sum(self.seg) > 0
                            else
                           self.return_0, ['P1_%s' % i for i in img_id]),
            'wquantile_5': (self.weighted_quantile_5 if np.sum(self.seg) > 0
                            else
                           self.return_0, ['P5_%s' % i for i in img_id]),
            'wquantile_25': (self.weighted_quantile_25 if np.sum(self.seg) >
                                                          0 else
                            self.return_0, ['P25_%s' % i for i in img_id]),
            'wquantile_50': (self.weighted_quantile_50 if np.sum(self.seg) > 0
                            else
                            self.return_0, ['P50_%s' % i for i in img_id]),
            'wquantile_75': (self.weighted_quantile_75 if np.sum(self.seg) >
                                                          0 else
                            self.return_0, ['P75_%s' % i for i in img_id]),
            'wquantile_95': (self.weighted_quantile_95 if np.sum(self.seg) >
                                                          0 else
                            self.return_0, ['P95_%s' % i for i in img_id]),
            'wquantile_99': (self.weighted_quantile_99 if np.sum(self.seg) >
                                                          0 else
                            self.return_0, ['P99_%s' % i for i in img_id]),

            'std': (self.std_ if np.sum(self.seg) > 0 else self.return_0,
                    ['STD_%s' % i for i in img_id]),
            'asm': (self.call_asm if np.sum(self.seg) > 0 else self.return_0,
                    ['asm%s' % i for i in img_id]),

            'contrast': (self.call_contrast if np.sum(self.seg) > 0 and
                         self.haralick_flag else self.return_0,
                         ['contrast%s' % i for i in img_id]),
            'correlation': (self.call_correlation if np.sum(self.seg) > 0 and
                            self.haralick_flag else self.return_0,
                            ['correlation%s' % i for i in img_id]),
            'sumsquare': (self.call_sum_square if np.sum(self.seg) > 0 and
                          self.haralick_flag else self.return_0,
                          ['sumsquare%s' % i for i in img_id]),
            'sum_average': (self.call_sum_average if np.sum(self.seg) > 0 and
                            self.haralick_flag else self.return_0,
                            ['sum_average%s' % i for i in img_id]),
            'idifferentmomment': (self.call_idifferent_moment if np.sum(
                self.seg) > 0 and self.haralick_flag else self.return_0,
                                  ['idifferentmomment%s' % i for i in img_id]),
            'sumentropy': (self.call_sum_entropy if np.sum(self.seg) > 0 and
                           self.haralick_flag else self.return_0,
                           ['sumentropy%s' % i for i in img_id]),
            'entropy': (self.call_entropy if np.sum(self.seg) > 0 and
                        self.haralick_flag else self.return_0,
                        ['entropy%s' % i for i in img_id]),
            'differencevariance': (self.call_difference_variance if
                                   np.sum(self.seg) > 0 and
                                   self.haralick_flag else self.return_0,
                                   ['differencevariance%s' % i for i in
                                    img_id]),
            'differenceentropy': (self.call_difference_entropy if
                                  np.sum(self.seg) > 0 and self.haralick_flag
                                  else self.return_0,
                                  ['differenceentropy%s' % i for i in img_id]),
            'sumvariance': (self.call_sum_variance if np.sum(self.seg) > 0 and
                            self.haralick_flag else self.return_0,
                            ['sumvariance%s' % i for i in img_id]),
            'imc1': (self.call_imc1 if np.sum(self.seg) > 0 and
                     self.haralick_flag else self.return_0, ['imc1%s' % i for
                                                             i in img_id]),
            'imc2': (self.call_imc2 if np.sum(self.seg) > 0 and
                     self.haralick_flag else self.return_0, ['imc2%s' % i for
                                                             i in img_id])

        }

    def __compute_mask(self):
        # TODO: check whether this works for probabilities type
        foreground_selector = np.where((self.seg > 0).reshape(-1))[0]
        probs = self.seg.reshape(-1)[foreground_selector]
        regions = np.zeros((foreground_selector.shape[0], self.img_channels))
        for i in np.arange(self.img_channels):
            regions[:, i] = self.img[..., 0, i].reshape(-1)[foreground_selector]
        return regions, probs

    def centre_of_mass(self):
        return np.mean(np.argwhere(self.seg > self.threshold), 0)

    def fractal_dimension(self):
        if np.sum(self.seg) < 6:
            return 0
        scales = np.logspace(0.05, 5, num=10, endpoint=False, base=2)
        #print(scales)
        offset_res = []
        for x_pad in range(0, 6):
            for y_pad in range(0, 6):
                for z_pad in range(0, 6):
                    seg_pad = np.pad(self.bin_seg, ((x_pad, x_pad), (y_pad,
                                                                    y_pad),
                                     (z_pad, z_pad)), mode='constant')
                    Ns_temp = RegionProperties.count_boxes_multiscales(seg_pad)
                    #print(x_pad, y_pad, z_pad)
                    offset_res.append(np.expand_dims(Ns_temp, 1))
        final_concat = np.concatenate(offset_res, 1)
        min_boxes = np.min(final_concat, 1)
        # linear fit, polynomial of degree 1
        coeffs = np.polyfit(np.log(scales), np.log(min_boxes), 1)
        print("The Hausdorff dimension is", -coeffs[0])
        # the fractal dimension is the OPPOSITE of the fitting coefficient
        return -1 * coeffs[0]

    def get_xyz_minmax(self):
        bord = RegionProperties.border_fromero(self.bin_seg)
        list_ind = np.asarray(np.where(bord > 0)).T
        min_ind = np.min(list_ind, 0)
        max_ind = np.max(list_ind, 0)
        return min_ind, max_ind

    @CacheFunctionOutput
    def convexhull_fromles(self):
        border_periv = RegionProperties.border_fromero(self.bin_seg)
        indices = np.asarray(np.where(border_periv)).T
        out_idx = indices
        if indices.shape[0] > 4:
            try:
                test_chi = CH(indices)
                deln = DL(test_chi.points[test_chi.vertices])
                idx = np.stack(np.indices(self.bin_seg.shape), axis=-1)
                out_idx = np.asarray(np.nonzero(deln.find_simplex(idx) + 1)).T
            except:
                out_idx = indices
        else:
            out_idx = indices
        max_shape = np.tile(np.expand_dims(np.asarray(border_periv.shape)-1,
                                           0), [np.asarray(out_idx).shape[0],1])
        out_idx = np.minimum(np.asarray(out_idx), max_shape)
        print(np.max(out_idx, 0))
        chi_temp = np.zeros(self.bin_seg.shape)
        chi_temp[list(out_idx.T)] = 1
        return chi_temp

    def circularity(self):
        bord_les = RegionProperties.border_fromero(self.bin_seg)
        return 4 * np.pi * np.sum(self.seg) / np.square(np.sum(bord_les))

    def solidity(self):
        ch_les = self.convexhull_fromles()
        return np.sum(self.bin_seg) / np.sum(ch_les)

    def contour_smoothness(self):
        ch_les = self.convexhull_fromles()
        border_les = RegionProperties.border_fromero(self.bin_seg)
        border_ch = RegionProperties.border_fromero(ch_les)
        return np.sum(border_ch) / np.sum(border_les)

    def balanced_rep(self):
        list_ind = np.asarray(np.where(self.bin_seg>0)).T
        std_ind = np.std(list_ind, 0)
        return np.sqrt(np.min(std_ind)/np.max(std_ind))

    @CacheFunctionOutput
    def ellipsoid_lambdas(self):
        idx = np.asarray(np.where(self.bin_seg>0)).T
        centre_of_mass = np.mean(idx, axis=0)

        idx_centred = idx - np.expand_dims(centre_of_mass,0)
        cov_idx = np.matmul(idx_centred.T, idx_centred)
        det = np.linalg.det(cov_idx)
        if det < 0.001:
            cov_idx += 0.00001 * np.eye(cov_idx.shape[0])

        u, s, v = np.linalg.svd(cov_idx)
        return np.sqrt(s)

    def ratio_eigen(self):
        s = self.ellipsoid_lambdas()
        return s[0]/s[1], s[0]/s[2], s[1]/s[2]

    def fractional_anisotropy(self):
        s = self.ellipsoid_lambdas()
        mean_eig = np.mean(s)
        fa = np.sqrt( np.sum(np.square(s-mean_eig)))/ np.sqrt(np.sum(
            np.square(s)))
        return np.sqrt(3.0/2.0) * fa

    @CacheFunctionOutput
    def weighted_quantile(self,
                          values_sorted=False, old_style=False):
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        list_weights = []
        list_values = []
        for m in range(0, self.img.shape[-1]):
            values = np.reshape(self.img[...,m], [-1])
            sample_weight = np.reshape(self.seg, [-1])
            ind = np.where(sample_weight > 0)
            values = values[ind]
            sample_weight = sample_weight[ind]

            if not values_sorted:
                sorter = np.argsort(values)
                values = values[sorter]
                sample_weight = sample_weight[sorter]

            weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
            if old_style:
                # To be convenient with numpy.percentile
                weighted_quantiles -= weighted_quantiles[0]
                weighted_quantiles /= weighted_quantiles[-1]
            else:
                weighted_quantiles /= np.sum(sample_weight)
            list_weights.append(weighted_quantiles)
            list_values.append(values)
        return list_weights, list_values

    def weighted_quantile_1(self):
        w = self.weighted_quantile()[0]
        v = self.weighted_quantile()[1]
        return [np.interp([0.01], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def weighted_quantile_5(self):
        return [np.interp([0.05], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def weighted_quantile_25(self):
        return [np.interp([0.25], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def weighted_quantile_50(self):
        return [np.interp([0.5], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def weighted_quantile_75(self):
        return [np.interp([0.75], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def weighted_quantile_95(self):
        return [np.interp([0.95], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def weighted_quantile_99(self):
        return [np.interp([0.99], w, v)[0] for (w, v) in zip(
            self.weighted_quantile()[0], self.weighted_quantile()[1])]

    def max_dist_com(self):
        border = RegionProperties.border_fromero(self.bin_seg)
        indices_border = np.asarray(np.where(border)).T
        com = self.centre_of_mass()
        dist_com = np.sum(np.square(indices_border - com) * np.square(self.pixdim))
        return np.sqrt(np.max(dist_com))


    # def grey_level_size_matrix(self):

    def volume(self):
        numb_seg = np.sum(self.seg)
        numb_seg_bin = np.sum(self.seg > 0)
        return numb_seg, numb_seg_bin, \
            numb_seg * self.vol_vox, numb_seg_bin * self.vol_vox

    def surface(self):
        border_seg = MorphologyOps(self.seg, self.neigh).border_map()
        numb_border_seg_bin = np.sum(border_seg > 0)
        numb_border_seg = np.sum(border_seg)
        return numb_border_seg, numb_border_seg_bin, \
            numb_border_seg * self.vol_vox, \
            numb_border_seg_bin * self.vol_vox

    def glcm(self):
        shifts = [[0, 0, 0],
                  [1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1],
                  [0, 0, -1],
                  [1, 1, 0],
                  [-1, -1, 0],
                  [-1, 1, 0],
                  [1, -1, 0],
                  [1, 1, 0],
                  [0, -1, -1],
                  [0, -1, 1],
                  [0, 1, -1],
                  [1, 0, 1],
                  [-1, 0, -1],
                  [-1, 0, 1],
                  [1, 0, -1],
                  [1, 1, 1],
                  [-1, 1, -1],
                  [-1, 1, 1],
                  [1, 1, -1],
                  [1, -1, 1],
                  [-1, -1, -1],
                  [-1, -1, 1],
                  [1, -1, -1]]
        bins = np.arange(0, self.bin)
        multi_mod_glcm = []
        if self.seg is None:
            return None
        for m in range(0, self.img.shape[4]):
            shifted_image = []
            for n in range(0, self.neigh+1):

                new_img = self.seg * self.img[:, :, :, 0, m]
                print(np.max(self.img), 'is max img')
                new_img = ndimage.shift(new_img, shifts[n], order=0)
                print(np.max(self.seg), 'is max shift')
                if np.count_nonzero(new_img) > 0:
                    flattened_new = new_img.flatten()
                    flattened_seg = self.seg.flatten()
                    affine = np.round(flattened_new*self.mul+self.trans)

                    select_new = np.digitize(affine[flattened_seg == 1], bins)
                    select_new[select_new >= self.bin] = self.bin-1
                    # print(np.max(select_new),' is max bin',np.max(affine))
                    shifted_image.append(select_new)
            glcm = np.zeros([self.bin, self.bin, self.neigh])
            for n in range(0, self.neigh):
                if len(shifted_image) > 0:
                    for i in range(0, shifted_image[0].size):
                        glcm[shifted_image[0][i], shifted_image[n+1][i], n] += 1
                    glcm[:, :, n] = glcm[:, :, n] / np.sum(glcm[:, :, n])
            multi_mod_glcm.append(glcm)
        return multi_mod_glcm

    def harilick_matrix(self):
        multi_mod_glcm = self.glcm()
        matrix_harilick = np.zeros([13, self.neigh, self.img_channels])
        if multi_mod_glcm is None:
            return np.average(matrix_harilick, axis=1)

        for i in range(0, self.img_channels):
            for j in range(0, self.neigh):
                matrix = multi_mod_glcm[i][..., j]
                harilick_vector = self.harilick(matrix)
                for index, elem in enumerate(harilick_vector):
                    matrix_harilick[index, j, i] = elem
        return np.average(matrix_harilick, axis=1)

    def call_asm(self):
        """
        Extract ASM data from Haralick matrix
        :return:
        """
        return self.harilick_m[0, :]

    def call_contrast(self):
        """
        Extract Contrast data from Haralick matrix
        :return:
        """
        return self.harilick_m[1, :]

    def call_correlation(self):
        """
        Extract Correlation result from Haralick matrix
        :return:
        """
        return self.harilick_m[2, :]

    def call_sum_square(self):
        """
        Extract Sum Square from Haralick matrix
        :return:
        """
        return self.harilick_m[3, :]

    def call_sum_average(self):
        """
        Extract Sum average from Haralick matrix
        :return:
        """
        return self.harilick_m[4, :]

    def call_idifferent_moment(self):
        """
        Extract IDifferent moment from Haralick matrix
        :return:
        """
        return self.harilick_m[5, :]

    def call_sum_entropy(self):
        """
        Extract Sum Entropy from Haralick matrix
        :return:
        """
        return self.harilick_m[6, :]

    def call_entropy(self):
        """
        Extract Entropy from Haralick matrix
        :return:
        """
        return self.harilick_m[7, :]

    def call_difference_variance(self):
        """
        Extract Difference Variance from Haralick Matrix
        :return:
        """
        return self.harilick_m[8, :]

    def call_difference_entropy(self):
        """
        Extract Difference Entropy from Haralick Matrix
        :return:
        """
        return self.harilick_m[9, :]

    def call_sum_variance(self):
        """
        Extract Sum Variance from Haralick Matrix
        :return:
        """
        return self.harilick_m[10, :]

    def call_imc1(self):
        """
        Extract IMC1 from Haralick Matrix
        :return:
        """
        return self.harilick_m[11, :]

    def call_imc2(self):
        """
        Extract IMC2 from Haralick Matrix
        :return:
        """
        return self.harilick_m[12, :]

    def fill_value(self):
        for key in self.m_dict:
            if key in self.measures:
                # print(key)
                result = self.m_dict[key][0]()
                # print(key, result)
                if np.sum(self.seg) == 0:
                    if not isinstance(result, (list, tuple, set, np.ndarray)):
                        result = float(result)
                    else:
                        result = map(float, result)

                if not isinstance(result, (list, tuple, set, np.ndarray)):
                    self.m_dict_result[key] = result
                else:
                    if isinstance(result, np.ndarray):
                        len_res = result.shape[0]
                    else:
                        len_res = len(result)
                    if len_res == len(self.img_id):
                        for (res, imgid) in zip(result, self.img_id):
                            key_new = key + '_' + imgid
                            self.m_dict_result[key_new] = res
                    else:
                        for d in range(len_res):
                            key_new = key + '_' + str(d)
                            self.m_dict_result[key_new] = result[d]

    @staticmethod
    def harilick(matrix):
        """
        Perform calculation of all haralick features from GLCM matrix and
        fill the associated matrix
        :param matrix: GLCM matrix
        :return:
        """
        vector_harilick = []
        asm = RegionProperties.angular_second_moment(matrix)
        contrast = RegionProperties.contrast(matrix)
        correlation = RegionProperties.correlation(matrix)
        sum_square = RegionProperties.sum_square_variance(matrix)
        sum_average = RegionProperties.sum_average(matrix)
        idifferentmoment = RegionProperties.inverse_difference_moment(matrix)
        sum_entropy = RegionProperties.sum_entropy(matrix)
        entropy = RegionProperties.entropy(matrix)
        differencevariance, differenceentropy = \
            RegionProperties.difference_variance_entropy(matrix)
        sum_variance = RegionProperties.sum_variance(matrix)
        imc1, imc2 = RegionProperties.information_measure_correlation(matrix)
        vector_harilick.append(asm)
        vector_harilick.append(contrast)
        vector_harilick.append(correlation)
        vector_harilick.append(sum_square)
        vector_harilick.append(sum_average)
        vector_harilick.append(idifferentmoment)
        vector_harilick.append(sum_entropy)
        vector_harilick.append(entropy)
        vector_harilick.append(differencevariance)
        vector_harilick.append(differenceentropy)
        vector_harilick.append(sum_variance)
        vector_harilick.append(imc1)
        vector_harilick.append(imc2)
        return vector_harilick

    @staticmethod
    def angular_second_moment(matrix):
        """
        Calculate ASM from GLCM matrix
        :param matrix:
        :return:
        """
        asm = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                asm += matrix[i, j] ** 2
        return asm

    @staticmethod
    def homogeneity(matrix):
        """
        Calculate Homogeneity from GLCM matrix
        :param matrix:
        :return:
        """
        homogeneity = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                homogeneity += matrix[i, j]/(1-abs(i-j))
        return homogeneity

    @staticmethod
    def energy(matrix):
        """
        Calculate Energy from GLCM matrix
        :param matrix:
        :return:
        """
        energy = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                energy += matrix[i, j] ** 2
        return energy

    @staticmethod
    def entropy(matrix):
        """
        Calculate Entropy from GLCM matrix
        :param matrix:
        :return:
        """
        entropy = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                if matrix[i, j] > 0:
                    entropy -= matrix[i, j] * math.log(matrix[i, j])
        return entropy

    @staticmethod
    def correlation(matrix):
        """
        Calculate Correlation from GLCM matrix
        :param matrix:
        :return:
        """
        range_values = np.arange(0, matrix.shape[0])
        matrix_range = np.tile(range_values, [matrix.shape[0], 1])
        mean_matrix = np.sum(matrix_range * matrix, axis=0)
        sd_matrix = np.sqrt(np.sum((matrix_range -
                                    np.tile(mean_matrix,
                                            [matrix.shape[0], 1]))**2 *
                                   matrix, axis=0))
        correlation = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                if sd_matrix[i] > 0 and sd_matrix[j] > 0:
                    correlation += (i*j*matrix[i, j]-mean_matrix[i] *
                                    mean_matrix[j]) / (sd_matrix[i] *
                                                       sd_matrix[j])
        return correlation

    @staticmethod
    def inverse_difference_moment(matrix):
        """
        Calculate I Difference moment from GLCM matrix
        :param matrix:
        :return:
        """
        idm = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                idm += 1.0 / (1 + (i-j)**2) * matrix[i, j]
        return idm

    @staticmethod
    def sum_average(matrix):
        """
        Calculate sum average from GLCM matrix
        :param matrix:
        :return:
        """
        sa = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                sa += (i+j) * matrix[i, j]
        return sa

    @staticmethod
    def sum_entropy(matrix):
        """
        Calculate sum entropy from GLCM matrix
        :param matrix:
        :return:
        """
        se = 0
        matrix_bis = np.zeros([2*matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[i+j] += matrix[i, j]
        for v in matrix_bis:
            if v > 0:
                se -= v*math.log(v)
        return se

    @staticmethod
    def sum_variance(matrix):
        """
        Calculate Sum variance from GLCM variance
        :param matrix:
        :return:
        """
        sv = 0
        se = RegionProperties.sum_entropy(matrix)
        matrix_bis = np.zeros([2 * matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[i + j] += matrix[i, j]
        for i in range(0, matrix_bis.size):
            sv += (i - se) ** 2 * matrix_bis[i]
        return sv

    @staticmethod
    def contrast(matrix):
        """
        Calculate Contrast from GLCM matrix
        :param matrix:
        :return:
        """
        contrast = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                contrast += (j-i)**2 * matrix[i, j]
        return contrast

    @staticmethod
    def difference_variance_entropy(matrix):
        """
        Calculate Difference Variance and Entropy from GLCM
        :param matrix:
        :return:
        """
        dv = 0
        de = 0
        matrix_bis = np.zeros([matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[abs(i-j)] += matrix[i, j]
        for i in range(0, matrix.shape[0]):
            dv += matrix_bis[i] * i ** 2
            if matrix_bis[i] > 0:
                de -= math.log(matrix_bis[i]) * matrix_bis[i]
        dv = 1.0/matrix_bis.shape[0] * dv - np.square(np.mean(matrix_bis))
        return dv, de

    @staticmethod
    def information_measure_correlation(matrix):
        """
        Calculate IMC from GCLM
        :param matrix:
        :return:
        """
        hxy = RegionProperties.entropy(matrix)
        sum_row = np.sum(matrix, axis=0)
        hxy_1 = 0
        hxy_2 = 0
        hx = 0
        for i in range(0, matrix.shape[0]):
            hx -= sum_row[i] * math.log(sum_row[i] + 0.001)
            for j in range(0, matrix.shape[0]):
                hxy_1 -= matrix[i, j] * math.log(sum_row[i]*sum_row[j] + 0.001)
                hxy_2 -= sum_row[i] * sum_row[j] * math.log(sum_row[i] *
                                                            sum_row[j] + 0.001)
        ic_1 = (hxy - hxy_1)/hx
        if hxy == 0:
            ic_2 = 0
        else:
            ic_2 = math.sqrt(1-math.exp(-2*np.abs((hxy_2-hxy))))
        return ic_1, ic_2

    @staticmethod
    def sum_square_variance(matrix):
        """
        Calculate sum square variance from GLCM matrix
        :param matrix:
        :return:
        """
        ssv = 0
        range_values = np.arange(0, matrix.shape[0])
        matrix_range = np.tile(range_values, [matrix.shape[0], 1])
        mean = np.average(matrix_range * matrix)
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                ssv += (i-mean) ** 2 * matrix[i, j]
        return ssv

    @staticmethod
    def border_fromero(maples):
        bord = maples - morph.binary_erosion(maples)
        print(np.sum(bord))
        return bord

    @staticmethod
    def count_boxes_multiscales(seg):
        indices = np.where(seg > 0)

        Lx = seg.shape[0]
        Ly = seg.shape[1]
        Lz = seg.shape[2]

        #print(Lx, Ly, Lz)
        pixels = np.asarray(indices).T
        #print(pixels.shape)

        # computing the fractal dimension
        # considering only scales in a logarithmic list
        scales = np.logspace(0.5, 5, num=10, endpoint=False, base=2)
        Ns = []
        # looping over several scales
        for scale in scales:
            # computing the histogram
            H, edges = np.histogramdd(indices, bins=(np.arange(0, Lx, scale),
                                                     np.arange(0, Ly, scale),
                                                     np.arange(0, Lz, scale)))
            Ns.append(np.sum(H > 0))
        return Ns

    def return_0(self):
        """
        Returns 0 function
        :return:
        """
        return [0] * self.img_channels

    def sav(self):
        """
        Calculate different modes of Surface area volume
        :return:
        """
        surf_numb, surf_bin, surf_vol, surf_volbin = self.surface()
        vol_numb, vol_bin, vol_vol, vol_volbin = self.volume()
        return surf_numb / vol_numb, surf_bin / vol_bin, surf_vol / vol_vol, \
            surf_volbin / vol_volbin

    def compactness(self):
        """
        Calculate different modes of compactness (number, binary, volume) based
        :return:
        """
        surf_n, surf_nb, surf_v, surf_vb = self.surface()
        vol_n, vol_nb, vol_v, vol_vb = self.volume()
        return np.power(surf_n, 1.5) / surf_n, np.power(surf_nb, 1.5) / \
            vol_nb, np.power(surf_v, 1.5) / vol_v, \
            np.power(surf_vb, 1.5) / vol_vb

    def min_(self):
        """
        Retun minimum over mask
        :return:
        """
        # print(ma.min(self.masked_img,0))
        return ma.min(self.masked_img, 0)

    def max_(self):
        """
        Return maximum over mask
        :return:
        """
        return ma.max(self.masked_img, 0)

    def weighted_mean_(self):
        """
        Return mean over probabilistic mask
        :return:
        """
        masked_seg = np.tile(self.masked_seg, [self.img_channels, 1]).T
        return ma.average(self.masked_img, axis=0, weights=masked_seg).flatten()

    def weighted_std_(self):
        """
        Return mean over probabilistic mask
        :return:
        """
        masked_seg = np.tile(self.masked_seg, [self.img_channels, 1]).T
        squared_img = self.masked_img * self.masked_img
        return ma.average(
            squared_img, axis=0, weights=masked_seg).flatten() - np.square(ma.average(self.masked_img, axis=0,
                           weights=masked_seg).flatten())

    def mean_(self):
        """
        Mean over mask
        :return:
        """
        return ma.mean(self.masked_img, 0)

    def skewness_(self):
        """
        Skewness of image over mask
        :return:
        """
        return mstats.skew(self.masked_img, 0)

    def std_(self):
        """
        Standard deviation of image over mask
        :return:
        """
        return ma.std(self.masked_img, 0)

    def kurtosis_(self):
        """
        Kurtosis of image over mask
        :return:
        """
        return mstats.kurtosis(self.masked_img, 0)

    def median_(self):
        """
        Median of image over mask
        :return:
        """
        return ma.median(self.masked_img, 0)

    def quantile_25(self):
        """
        1st quartile of image over mask
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=0.25, axis=0).flatten()

    def quantile_75(self):
        """
        3rd quartile of image over mask
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=0.75, axis=0).flatten()

    def quantile_5(self):
        """
        5th percentile of image over mask
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=0.05, axis=0).flatten()

    def quantile_95(self):
        """
        95th percentile of image over mask
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=0.95, axis=0).flatten()

    def quantile_1(self):
        """
        1st percentile of image over mask
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=0.01, axis=0).flatten()

    def quantile_99(self):
        """
        99th percentile of image over mask
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=0.99, axis=0).flatten()

    def quantile(self, value):
        """
        Quantile of value value (between 0 and 1) of image over mask
        :param value:
        :return:
        """
        return mstats.mquantiles(self.masked_img, prob=value, axis=0).flatten()

    def header_str(self):
        """
        Create header string according to measures used
        :return:
        """
        result_str = [j for i in self.measures for j in self.m_dict[i][1]]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:4f}'):
        """
        Transformation to string of obtained results
        :param fmt:
        :return:
        """
        result_str = ""
        for i in self.measures:
            print(i, self.m_dict[i])
            result = self.m_dict[i][0]()
            if isinstance(result, (list, tuple, np.ndarray)):
                for j in self.m_dict[i][0]():
                    print(j)
                    try:
                        j = float(np.nan_to_num(j))
                        fmt = fmt
                    except ValueError:
                        j = j
                        fmt = '{!s:4s}'
                    try:

                        result_str += ',' + fmt.format(j)
                    except ValueError:  # some functions give strings e.g., "--"
                        print(i, j)
                        result_str += ',{}'.format(j)
            else:
                j = result
                try:
                    j = float(np.nan_to_num(j))
                    fmt = fmt
                except ValueError:
                    j = j
                    fmt = '{!s:4s}'
                try:

                    result_str += ',' + fmt.format(j)
                except ValueError:  # some functions give strings e.g., "--"
                    print(i, j)
                    result_str += ',{}'.format(j)
        return result_str


# class MorphologyOps(object):
#     '''
#     Class that performs the morphological operations needed to get notably
#     connected component. To be used in the evaluation
#     '''
#
#     def __init__(self, binary_img, neigh):
#         self.binary_map = np.asarray(binary_img, dtype=np.int8)
#         self.neigh = neigh
#
#     def border_map(self):
#         '''
#         Creates the border for a 3D image
#         :return:
#         '''
#         west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
#         east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
#         north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
#         south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
#         top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
#         bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
#         cumulative = west + east + north + south + top + bottom
#         border = ((cumulative < 6) * self.binary_map) == 1
#         return border
#
#     def foreground_component(self):
#         return ndimage.label(self.binary_map)
