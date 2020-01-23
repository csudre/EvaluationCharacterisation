from __future__ import absolute_import, print_function

import numpy as np
from scipy import ndimage
import scipy.spatial.distance as dist
from scipy.linalg import svd
from scipy.ndimage import measurements as meas
import numpy.ma as ma
from skimage.morphology import skeletonize_3d as sk3d
from functools import partial
from scipy.stats import mstats as mstats
from .morphology import MorphologyOps


class CacheFunctionOutput(object):
    """
    this provides a decorator to cache function outputs
    to avoid repeating some heavy function computations
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return partial(self, obj)  # to remember func as self.func

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*args, **kw)
        return value


class PairwiseMeasures(object):
    def __init__(self, seg_img, ref_img,
                 measures=None, num_neighbors=8, pixdim=[1, 1, 1],
                 empty=False, list_labels=None):

        self.m_dict = {
            'green volume': (self.n_pos_ref, 'Volume_(Green)'),
            'red volume': (self.n_pos_seg, 'Volume_(Red)'),
            'n_intersection': (self.n_intersection, 'Intersection'),
            'n_union': (self.n_union, 'Union'),
            'IoU': (self.intersection_over_union, 'IoU'),
            'coverage': (self.overlap, 'Overlap'),
            'vol_diff': (self.vol_diff, 'VolDiff'),
            'ave_dist': (self.measured_average_distance, 'AveDist'),
            'haus_dist': (self.measured_hausdorff_distance, 'HausDist'),
            'haus_dist95': (self.measured_hausdorff_distance_95, 'HausDist95'),
            'com_dist': (self.com_dist, 'COM distance'),
            'com_ref': (self.com_ref, 'COM red'),
            'com_seg': (self.com_seg, 'COM green')
        }
        self.seg = seg_img
        self.ref = ref_img
        self.seg_bin = np.where(seg_img > 0, np.ones_like(seg_img),
                                np.zeros_like(seg_img))
        self.ref_bin = np.where(ref_img > 0, np.ones_like(ref_img),
                                np.zeros_like(ref_img))
        self.list_labels = list_labels
        self.flag_empty = empty
        self.measures = measures if measures is not None else self.m_dict
        self.neigh = num_neighbors
        self.pixdim = pixdim
        self.m_dict_result = {}

    def __fp_map(self):
        """
        This function calculates the false positive map
        :return: FP map
        """
        return np.asarray((self.seg - self.ref) > 0.0, dtype=np.float32)

    def __fn_map(self):
        """
        This function calculates the false negative map
        :return: FN map
        """
        return np.asarray((self.ref - self.seg) > 0.0, dtype=np.float32)

    def __tp_map(self):
        """
        This function calculates the true positive map
        :return: TP map
        """
        return np.asarray((self.ref + self.seg) > 1.0, dtype=np.float32)

    def __tn_map(self):
        """
        This function calculates the true negative map
        :return: TN map
        """
        return np.asarray((self.ref + self.seg) < 0.5, dtype=np.float32)

    def __union_map(self):
        """
        This function calculates the union map between segmentation and
        reference image
        :return: union map
        """
        return np.asarray((self.ref + self.seg) > 0.5, dtype=np.float32)

    def __intersection_map(self):
        """
        This function calculates the intersection between segmentation and
        reference image
        :return: intersection map
        """
        return np.multiply(self.ref_bin, self.seg_bin)

    @CacheFunctionOutput
    def n_pos_ref(self):
        return np.sum(self.ref)

    @CacheFunctionOutput
    def n_neg_ref(self):
        return np.sum(1 - self.ref)

    @CacheFunctionOutput
    def n_pos_seg(self):
        return np.sum(self.seg)

    @CacheFunctionOutput
    def n_neg_seg(self):
        return np.sum(1 - self.seg)

    @CacheFunctionOutput
    def fp(self):
        return np.sum(self.__fp_map())

    @CacheFunctionOutput
    def fn(self):
        return np.sum(self.__fn_map())

    @CacheFunctionOutput
    def tp(self):
        return np.sum(self.__tp_map())

    @CacheFunctionOutput
    def tn(self):
        return np.sum(self.__tn_map())

    @CacheFunctionOutput
    def n_intersection(self):
        return np.sum(self.__intersection_map())

    @CacheFunctionOutput
    def n_union(self):
        return np.sum(self.__union_map())

    def overlap(self):
        return np.sum(self.seg)/np.sum(self.ref) * 100

    def intersection_over_union(self):
        """
        This function the intersection over union ratio - Definition of
        jaccard coefficient
        :return:
        """
        return self.n_intersection() / self.n_union()

    def com_dist(self):
        """
        This function calculates the euclidean distance between the centres
        of mass of the reference and segmentation.
        :return:
        """
        if self.flag_empty:
            return -1
        else:
            com_ref = ndimage.center_of_mass(self.ref)
            com_seg = ndimage.center_of_mass(self.seg)
            com_dist = np.sqrt(np.dot(np.square(np.asarray(com_ref) -
                                                np.asarray(com_seg)), np.square(
                                                self.pixdim)))
            return com_dist

    def com_ref(self):
        """
        This function calculates the centre of mass of the reference
        segmentation
        :return:
        """
        return ndimage.center_of_mass(self.ref)

    def com_seg(self):
        """
        This functions provides the centre of mass of the segmented element
        :return:
        """
        if self.flag_empty:
            return -1
        else:
            return ndimage.center_of_mass(self.seg)

    def list_labels(self):
        if self.list_labels is None:
            return ()
        return tuple(np.unique(self.list_labels))

    def vol_diff(self):
        """
        This function calculates the ratio of difference in volume between
        the reference and segmentation images.
        :return: vol_diff
        """
        return np.abs(self.n_pos_ref() - self.n_pos_seg()) / self.n_pos_ref()

    # @CacheFunctionOutput
    # def _boundaries_dist_mat(self):
    #     dist = DistanceMetric.get_metric('euclidean')
    #     border_ref = MorphologyOps(self.ref, self.neigh).border_map()
    #     border_seg = MorphologyOps(self.seg, self.neigh).border_map()
    #     coord_ref = np.multiply(np.argwhere(border_ref > 0), self.pixdim)
    #     coord_seg = np.multiply(np.argwhere(border_seg > 0), self.pixdim)
    #     pairwise_dist = dist.pairwise(coord_ref, coord_seg)
    #     return pairwise_dist

    @CacheFunctionOutput
    def border_distance(self):
        """
        This functions determines the map of distance from the borders of the
        segmentation and the reference and the border maps themselves
        :return: distance_border_ref, distance_border_seg, border_ref,
        border_seg
        """
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        border_seg = MorphologyOps(self.seg,
                                   self.neigh).border_map()
        oppose_ref = 1 - self.ref/np.where(self.ref == 0, np.ones_like(
            self.ref), self.ref)
        oppose_seg = 1 - self.seg/np.where(self.seg == 0, np.ones_like(
            self.seg), self.seg)
        distance_ref = ndimage.distance_transform_edt(oppose_ref)
        distance_seg = ndimage.distance_transform_edt(oppose_seg)
        distance_border_seg = border_ref * distance_seg
        distance_border_ref = border_seg * distance_ref
        return distance_border_ref, distance_border_seg, border_ref, border_seg

    def measured_distance(self):
        """
        This functions calculates the average symmetric distance and the
        hausdorff distance between a segmentation and a reference image
        :return: hausdorff distance and average symmetric distance
        """
        ref_border_dist, seg_border_dist, ref_border, \
            seg_border = self.border_distance()
        average_distance = (np.sum(ref_border_dist) + np.sum(
            seg_border_dist)) / (np.sum(seg_border+ref_border))
        hausdorff_distance = np.max([np.max(ref_border_dist), np.max(
            seg_border_dist)])
        hausdorff_distance_95 = np.max([np.percentile(ref_border_dist[
                                                       self.ref+self.seg > 0],
                                                      q=95),
                                        np.percentile(
            seg_border_dist[self.ref+self.seg > 0], q=95)])
        return hausdorff_distance, average_distance, hausdorff_distance_95

    def measured_average_distance(self):
        """
        This function returns only the average distance when calculating the
        distances between segmentation and reference
        :return:
        """
        return self.measured_distance()[1]

    def measured_hausdorff_distance(self):
        """
        This function returns only the hausdorff distance when calculated the
        distances between segmentation and reference
        :return:
        """
        return self.measured_distance()[0]

    def measured_hausdorff_distance_95(self):
        return self.measured_distance()[2]

    # def average_distance(self):
    #     pairwise_dist = self._boundaries_dist_mat()
    #     return (np.sum(np.min(pairwise_dist, 0)) + \
    #             np.sum(np.min(pairwise_dist, 1))) / \
    #            (np.sum(self.ref + self.seg))
    #
    # def hausdorff_distance(self):
    #     pairwise_dist = self._boundaries_dist_mat()
    #     return np.max((np.max(np.min(pairwise_dist, 0)),
    #                    np.max(np.min(pairwise_dist, 1))))
    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str = ',' + ','.join(result_str)
        return result_str

    def fill_value(self):
        for key in self.m_dict:
            result = self.m_dict[key][0]()
            if not isinstance(result, (list, tuple, set, np.ndarray)):
                self.m_dict_result[key] = result
            else:
                for d in range(len(result)):
                    key_new = key + '_' + str(d)
                    self.m_dict_result[key_new] = result[d]

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        list_space = ['com_ref', 'com_seg', 'list_labels']
        for key in self.measures:
            result = self.m_dict[key][0]()
            if key in list_space:
                result_str += ' '.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            else:
                result_str += ','.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma


class RegionProperties(object):
    def __init__(self, seg, img=None, measures=None,
                 num_neighbors=18, threshold=0, pixdim=None):
        if pixdim is None:
            pixdim = [1, 1, 1]
        self.seg = seg
        self.order = [1, 0, 2]
        self.voxel_size = np.prod(pixdim)
        self.pixdim = pixdim
        if img is None:
            self.img = seg
        else:
            self.img = img
        self.img_channels = self.img.shape[4] if self.img.ndim >= 4 else 1
        for i in range(self.img.ndim, 5):
            self.img = np.expand_dims(self.img, -1)
        self.bin = bin
        self.threshold = threshold
        if self.seg is not None:
            self.masked_img, self.masked_seg = self.__compute_mask()
        else:
            print("no mask")
        self.neigh = num_neighbors
        self.connect = MorphologyOps(self.binarise(),
                                     self.neigh).connect()[0]
        # self.glszm = self.grey_level_size_matrix()
        self.m_dict = {
            'centre of mass': (self.centre_of_mass, ['CoMx',
                                                     'CoMy',
                                                     'CoMz']),
            'centre_abs': (self.centre_abs, ['Truex, Truey, Truez']),
            'volume': (self.volume,
                       ['NVoxels', 'NVolume']),
            'fragmentation': (self.fragmentation, ['Fragmentation']),
            'mean_intensity': (self.mean_int, ['MeanIntensity']),
            'surface': (self.surface, ['NSurface', 'Nfaces_surf',
                                       'NSurf_ext', 'Nfaces_ext']),
            'surface_dil': (self.surface_dil, ['surf_dil', 'surf_ero']),
            'surface volume ratio': (self.sav, ['sav_dil', 'sav_ero']),
            'compactness': (self.compactness, ['CompactNumbDil'
                                               ]),
            'eigen': (self.eigen, ['eigenvalues']),
            'std': (self.std_values, ['std']),
            'quantiles': (self.quantile_values, ['quantiles']),
            'bounds': (self.bounds, ['bounds']),
            'cc': (self.connect_cc, ['N_CC']),
            'cc_dist': (self.dist_cc, ['MeanDistCC']),
            'cc_size': (self.cc_size, ['MinSize', 'MaxSize', 'MeanSize']),
            'max_extent': (self.max_extent, ['MaxExtent']),
            'shape_factor': (self.shape_factor, ['ShapeFactor',
                                                 'shapefactor_surfcount']),
            'skeleton_length': (self.skeleton_length, ['SkeletonLength'])
        }
        self.measures = measures if measures is not None else self.m_dict
        self.m_dict_result = {}

    def binarise(self):
        binary_img = np.where(self.seg > 0, np.ones_like(self.seg),
                              np.zeros_like(self.seg))
        return binary_img

    def __compute_mask(self):
        # TODO: check whether this works for probabilities type
        foreground_selector = np.where((self.seg > 0).reshape(-1))[0]
        probs = self.seg.reshape(-1)[foreground_selector]
        regions = np.zeros((foreground_selector.shape[0], self.img_channels))
        for i in np.arange(self.img_channels):
            regions[:, i] = self.img[..., 0, i].reshape(-1)[foreground_selector]
        return regions, probs

    def shape_factor(self):
        binarised = self.binarise()
        vol = np.sum(binarised)
        if vol == 0:
            return 0, 0, 0
        radius = (vol * 0.75 * np.pi) ** (1.0/3.0)
        surf_sphere = 4 * np.pi * radius * radius
        surf_map, count_surf, _, _ = MorphologyOps(binarised, 6).border_surface_measures()
        count_fin = np.where(count_surf > 0, 6-count_surf, count_surf)
        count_final_surf = np.sum(count_fin)
        vol_change = np.pi ** (1/3) * (6*vol) ** (2/3)
        return surf_sphere / np.sum(surf_map), surf_sphere / \
            count_final_surf, vol_change/np.sum(surf_map)

    def skeleton_length(self):
        return np.sum(MorphologyOps(self.binarise(), 6).skeleton_map())

    def centre_of_mass(self):
        return list(np.mean(np.argwhere(self.seg > self.threshold), 0))

    def centre_abs(self):
        mean_centre = np.mean(np.argwhere(self.seg > self.threshold), 0)
        mean_centre_new = mean_centre[self.order]
        return list(mean_centre_new * self.pixdim)

    # def grey_level_size_matrix(self):

    def volume(self):
        # numb_seg = np.sum(self.seg)
        numb_seg_bin = np.sum(self.seg > 0)
        return numb_seg_bin, numb_seg_bin*self.voxel_size

    def surface(self):

        border_seg, count_surf, border_ext, count_surf_ext = MorphologyOps(
            self.binarise(), self.neigh, pixdim=self.pixdim).border_surface_measures()
        numb_border_seg = np.sum(border_seg)
        count_surfaces = np.where(count_surf > 0, 6-count_surf, count_surf)
        numb_border_ext = np.sum(border_ext)
        count_surfaces_ext = np.where(count_surf_ext > 0, 6 - count_surf_ext,
                                      count_surf_ext)
        return numb_border_seg, np.sum(count_surfaces), numb_border_ext, \
            np.sum(count_surfaces_ext)

    def surface_dil(self):
        return np.sum(MorphologyOps(self.binarise(), self.neigh).dilate() -
                      self.binarise()), \
            np.sum(self.binarise() - MorphologyOps(self.binarise(),
                                                   self.neigh).erode())

    def cc_size(self):
        if self.connect is None:
            self.connect = MorphologyOps(self.binarise(),
                                         self.neigh).connect()[0]
        min_size = 100000
        max_size = 0
        nf = np.max(self.connect)
        for l in range(1, nf+1):
            bin_label = np.where(self.connect == l, np.ones_like(self.connect),
                                 np.zeros_like(self.connect))
            if np.sum(bin_label) > max_size:
                max_size = np.sum(bin_label)
            if np.sum(bin_label) < min_size:
                min_size = np.sum(bin_label)
        return min_size, max_size, np.sum(self.binarise())/nf

    def connect_cc(self):
        if self.connect is None:
            self.connect = MorphologyOps(self.binarise(),
                                         self.neigh).connect()[0]
        return np.max(self.connect)

    def fragmentation(self):
        if self.connect is None:
            self.connect = MorphologyOps(self.binarise(),
                                         self.neigh).connect()[0]
        return 1 - 1.0/(np.max(self.connect)+0.000001)

    def dist_cc(self):
        if self.connect is None:
            self.connect = MorphologyOps(self.binarise(),
                                         self.neigh).connect()[0]
        connected, numb_frac = self.connect, np.max(self.connect)
        if numb_frac == 1:
            return 0
        else:
            dist_array = []
            size_array = []
            for label in range(numb_frac):
                indices_l = np.asarray(np.where(connected == label+1)).T
                for j in range(label+1, numb_frac):
                    indices_j = np.asarray(np.where(connected == j + 1)).T
                    size_array.append(indices_j.shape[0] + indices_l.shape[0])
                    dist_array.append(np.mean(dist.cdist(indices_l,
                                                         indices_j,
                                                         'wminkowski',
                                                         p= 2,
                                                         w=self.pixdim)))
            return np.sum(np.asarray(dist_array) * np.asarray(size_array) /
                          np.sum(np.asarray(size_array)))

    def bounds(self):
        indices = np.asarray(np.where(self.binarise() == 1)).T
        if np.sum(self.seg) == 0:
            return [0, ] * self.img.ndim
        min_bound = np.min(indices, 0)
        max_bound = np.max(indices, 0)
        return max_bound - min_bound

    def std_values(self):
        mask = 1 - self.binarise()

        data_masked = ma.array(self.img, mask=mask)
        return ma.std(data_masked)

    def quantile_values(self):
        mask = 1 - self.binarise()
        data_masked = ma.array(self.img, mask=mask)
        values = mstats.mquantiles(np.reshape(data_masked, [-1]), [0.05, 0.95],
                                   axis=0)
        return ma.min(data_masked), ma.max(data_masked), values[0], values[1]

    def max_extent(self):
        if np.sum(self.binarise()) < 2:
            return 0
        indices = np.asarray(np.where(self.binarise() == 1)).T
        pdist = dist.pdist(indices, 'wminkowski', p=2, w=self.pixdim)
        return np.max(pdist)

    def sav(self):
        surf_dil, surf_ero = self.surface_dil()
        vol = self.volume()
        return surf_dil / vol[0], surf_ero / vol[0]

    def compactness(self):
        surf_dil, surf_ero = self.surface_dil()
        vol = self.volume()
        return np.power(surf_dil, 1.5) / vol[0], np.power(surf_ero, 1.5) / \
            vol[0]

    def eigen(self):
        idx = np.where(self.seg > 0)
        try:
            _, eig_values, _ = svd(np.asarray(idx) - np.mean(np.asarray(idx),
                                                             1))
        except ValueError:
            eig_values = [0, ] * len(idx)
        return eig_values

    def mean_int(self):
        return np.sum(self.img)/np.sum(self.binarise())

    def header_str(self):
        result_str = [j for i in self.measures for j in self.m_dict[i][1]]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:4f}'):
        result_str = ""
        for i in self.measures:
            # print(i, self.m_dict[i])
            for j in self.m_dict[i][0]():
                # print(j)
                try:
                    j = float(np.nan_to_num(j))
                    fmt = fmt
                except ValueError:
                    j = j
                    fmt = '{!s:4s}'
                try:

                    result_str += ',' + fmt.format(j)
                except ValueError:  # some functions give strings e.g., "--"
                    # print(i, j)
                    result_str += ',{}'.format(j)
        return result_str

    def fill_value(self):
        for key in self.m_dict:
            # print(key)
            result = self.m_dict[key][0]()
            # print(key, result)
            if np.sum(self.seg) == 0:
                if not isinstance(result, (list, tuple, set)):
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
                for d in range(len_res):
                    key_new = key + '_' + str(d)
                    self.m_dict_result[key_new] = result[d]

