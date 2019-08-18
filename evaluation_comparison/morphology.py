
import numpy as np
from skimage import measure
from scipy import ndimage
from skimage.morphology import skeletonize_3d as sk3d


class MorphologyOps(object):
    '''
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    '''

    def __init__(self, binary_img, neigh, pixdim=None):
        if pixdim is None:
            pixdim = [1, 1, 1]
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh
        self.structure = self.create_connective_support()
        self.pixdim = pixdim

    def create_connective_support(self):
        connection = 2
        if self.neigh == 6 or self.neigh > 10:
            dim = 3
        else:
            dim = 2
        if self.neigh < 8:
            connection = 1
        elif self.neigh > 20:
            connection = 3
        init = np.ones([3] * dim)
        results = np.zeros([3] * dim)
        centre = [1] * dim
        idx = np.asarray(np.where(init > 0)).T
        diff_to_centre = idx - np.tile(centre, [idx.shape[0], 1])
        sum_diff_to_centre = np.sum(np.abs(diff_to_centre), axis=1)
        idx_chosen = np.asarray(np.where(sum_diff_to_centre <= connection)).T
        np.put(results, np.squeeze(idx_chosen)[:], 1)
        # print(np.sum(results))
        return results

    def skeleton_map(self):
        skeleton = sk3d(self.binary_map)
        return skeleton

    def border_surface_measures(self):
        connectivity = self.create_connective_support()
        dil = ndimage.morphology.binary_dilation(self.binary_map,
                                                 structure=connectivity,
                                                 iterations=1)
        ero = ndimage.morphology.binary_erosion(self.binary_map,
                                                 structure=connectivity,
                                                 iterations=1)
        internal_border = self.binary_map - ero
        external_border = dil - self.binary_map

        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)

        count_west_int = np.sum(np.where(np.logical_and(internal_border==1,
                                                 west==0), np.ones_like(dil),
                                         np.zeros_like(dil)))
        count_west_ext = np.sum(np.where(np.logical_and(external_border==1,
                                                 west==0), np.ones_like(dil),
                                         np.zeros_like(dil)))

        count_east_int = np.sum(np.where(np.logical_and(internal_border == 1,
                                                 east == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))
        count_east_ext = np.sum(np.where(np.logical_and(external_border == 1,
                                                 east == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))

        count_south_int = np.sum(np.where(np.logical_and(internal_border == 1,
                                                 south == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))
        count_south_ext = np.sum(np.where(np.logical_and(external_border == 1,
                                                 south == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))

        count_north_int = np.sum(np.where(np.logical_and(internal_border == 1,
                                                 north == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))
        count_north_ext = np.sum(np.where(np.logical_and(external_border == 1,
                                                 north == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))

        count_top_int = np.sum(np.where(np.logical_and(internal_border == 1,
                                                 top == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))
        count_top_ext = np.sum(np.where(np.logical_and(external_border == 1,
                                                 top == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))

        count_bottom_int = np.sum(np.where(np.logical_and(internal_border == 1,
                                                 bottom == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))
        count_bottom_ext = np.sum(np.where(np.logical_and(external_border == 1,
                                                 bottom == 0), np.ones_like(dil),
                                         np.zeros_like(dil)))

        top_area = self.pixdim[0] * self.pixdim[1]
        west_area = self.pixdim[0] * self.pixdim[2]
        south_area = self.pixdim[1] * self.pixdim[2]

        intbord_surf = count_east_int * west_area + count_west_int * \
                       west_area + count_bottom_int * top_area + \
                       count_top_int * top_area + count_south_int * \
                       south_area + count_north_int * south_area

        extbord_surf = count_east_ext * west_area + count_west_ext * \
                       west_area + count_bottom_ext * top_area + \
                       count_top_ext * top_area + count_south_ext * \
                       south_area + count_north_ext * south_area

        return np.sum(internal_border), intbord_surf, np.sum(
            external_border), extbord_surf




    def border_map(self):
        '''
        Creates the border for a 3D image
        :return:
        '''
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return ndimage.label(self.binary_map)

    def label_binary(self):
        blobs_labels = measure.label(self.binary_map, background=0)
        return blobs_labels

    def dilate(self, numb_dil=1):
        return ndimage.morphology.binary_dilation(self.binary_map,
                                                  structure=self.structure,
                                                  iterations=numb_dil)

    def erode(self, numb_ero=1):
        return ndimage.morphology.binary_erosion(self.binary_map,
                                                 self.structure, numb_ero)

    def connect(self):
        return measure.label(self.binary_map, connectivity=3)

    @staticmethod
    def oppose(img):
        bin_img = np.where(img > 0, np.ones_like(img), np.zeros_like(img))
        opp_img = 1-bin_img
        return opp_img

