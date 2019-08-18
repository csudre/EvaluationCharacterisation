from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.ma as ma
import math
import scipy.stats.mstats as mstats
import scipy.ndimage as ndimage
from .morphology import MorphologyOps
# from niftynet.utilities.misc_common import MorphologyOps, CacheFunctionOutput
LIST_HARALICK = ['asm', 'contrast', 'correlation', 'sumsquare',
                 'sum_average', 'idifferentmomment', 'sumentropy', 'entropy',
                 'differencevariance', 'sumvariance', 'differenceentropy',
                 'imc1', 'imc2']


class RegionProperties(object):
    def __init__(self, seg, img, measures,
                 num_neighbors=6, threshold=0, pixdim=None, bin_numb=100,
                 bin_glszm=32, mul=10, trans=50):
        if pixdim is None:
            pixdim = [1, 1, 1]
        self.seg = seg
        self.bin = bin_numb
        self.bin_glszm = bin_glszm
        self.mul = mul  # 10 for Norm 5 for Z 50 for Stacked
        self.trans = trans  # 50 for all
        self.img = img
        self.measures = measures

        self.pixdim = pixdim
        self.threshold = threshold
        if any(LIST_HARALICK) in measures or 'asm' in measures:
            self.haralick_flag = True
        else:
            self.haralick_flag = False
        print("Harilick flag", self.haralick_flag)

        self.vol_vox = np.prod(pixdim)
        self.img_channels = self.img.shape[4] if img.ndim >= 4 else 1
        img_id = range(0, self.img_channels)
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
            'mean': (self.mean_, ['Mean_%d' % i for i in img_id]),
            'weighted_mean': (self.weighted_mean_,
                              ['Weighted_mean_%d' % i for i in img_id]),
            'median': (self.median_, ['Median_%d' % i for i in img_id]),
            'skewness': (self.skewness_, ['Skewness_%d' % i for i in img_id]),
            'kurtosis': (self.kurtosis_, ['Kurtosis_%d' % i for i in img_id]),
            'min': (self.min_ if np.sum(self.seg) > 0 else
                    self.return_0, ['Min_%d' % i for i in img_id]),
            'max': (self.max_ if np.sum(self.seg) > 0 else
                    self.return_0, ['Max_%d' % i for i in img_id]),
            'quantile_1': (self.quantile_1 if np.sum(self.seg) > 0 else
                           self.return_0, ['P1_%d' % i for i in img_id]),
            'quantile_5': (self.quantile_5 if np.sum(self.seg) > 0 else
                           self.return_0, ['P5_%d' % i for i in img_id]),
            'quantile_25': (self.quantile_25 if np.sum(self.seg) > 0 else
                            self.return_0, ['P25_%d' % i for i in img_id]),
            'quantile_50': (self.median_ if np.sum(self.seg) > 0 else
                            self.return_0, ['P50_%d' % i for i in img_id]),
            'quantile_75': (self.quantile_75 if np.sum(self.seg) > 0 else
                            self.return_0, ['P75_%d' % i for i in img_id]),
            'quantile_95': (self.quantile_95 if np.sum(self.seg) > 0 else
                            self.return_0, ['P95_%d' % i for i in img_id]),
            'quantile_99': (self.quantile_99 if np.sum(self.seg) > 0 else
                            self.return_0, ['P99_%d' % i for i in img_id]),
            'std': (self.std_ if np.sum(self.seg) > 0 else self.return_0,
                    ['STD_%d' % i for i in img_id]),
            'asm': (self.call_asm if np.sum(self.seg) > 0 else self.return_0,
                    ['asm%d' % i for i in img_id]),

            'contrast': (self.call_contrast if np.sum(self.seg) > 0 and
                         self.haralick_flag else self.return_0,
                         ['contrast%d' % i for i in img_id]),
            'correlation': (self.call_correlation if np.sum(self.seg) > 0 and
                            self.haralick_flag else self.return_0,
                            ['correlation%d' % i for i in img_id]),
            'sumsquare': (self.call_sum_square if np.sum(self.seg) > 0 and
                          self.haralick_flag else self.return_0,
                          ['sumsquare%d' % i for i in img_id]),
            'sum_average': (self.call_sum_average if np.sum(self.seg) > 0 and
                            self.haralick_flag else self.return_0,
                            ['sum_average%d' % i for i in img_id]),
            'idifferentmomment': (self.call_idifferent_moment if np.sum(
                self.seg) > 0 and self.haralick_flag else self.return_0,
                                  ['idifferentmomment%d' % i for i in img_id]),
            'sumentropy': (self.call_sum_entropy if np.sum(self.seg) > 0 and
                           self.haralick_flag else self.return_0,
                           ['sumentropy%d' % i for i in img_id]),
            'entropy': (self.call_entropy if np.sum(self.seg) > 0 and
                        self.haralick_flag else self.return_0,
                        ['entropy%d' % i for i in img_id]),
            'differencevariance': (self.call_difference_variance if
                                   np.sum(self.seg) > 0 and
                                   self.haralick_flag else self.return_0,
                                   ['differencevariance%d' % i for i in
                                    img_id]),
            'differenceentropy': (self.call_difference_entropy if
                                  np.sum(self.seg) > 0 and self.haralick_flag
                                  else self.return_0,
                                  ['differenceentropy%d' % i for i in img_id]),
            'sumvariance': (self.call_sum_variance if np.sum(self.seg) > 0 and
                            self.haralick_flag else self.return_0,
                            ['sumvariance%d' % i for i in img_id]),
            'imc1': (self.call_imc1 if np.sum(self.seg) > 0 and
                     self.haralick_flag else self.return_0, ['imc1%d' % i for
                                                             i in img_id]),
            'imc2': (self.call_imc2 if np.sum(self.seg) > 0 and
                     self.haralick_flag else self.return_0, ['imc2%d' % i for
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
            ic_2 = math.sqrt(1-math.exp(-2*(hxy_2-hxy)))
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
