from scipy import ndimage
import numpy as np
from skimage import measure
from evaluation_comparison.morphology import MorphologyOps


class CompareDistances():
    def __init__(self, seg, ref,  measures, neigh=6, pixdim=None):
        if pixdim is None:
            pixdim = [1, 1, 1]
        self.seg = seg
        self.ref = ref
        self.neigh = neigh
        self.measures = measures
        self.pixdim = pixdim
        self.m_dict = {
            'distances': (self.measured_distance, ['HD', 'AvD', 'HD_95'])}

    def border_distance(self):
        """
        This functions determines the map of distance from the borders of the
        segmentation and the reference and the border maps themselves
        :return: distance_border_ref, distance_border_seg, border_ref,
        border_seg
        """
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        border_seg = MorphologyOps(self.seg, self.neigh).border_map()
        oppose_ref = 1 - self.ref
        oppose_seg = 1 - self.seg
        distance_ref = ndimage.distance_transform_edt(oppose_ref,
                                                      sampling=self.pixdim)
        distance_seg = ndimage.distance_transform_edt(oppose_seg,
                                                      sampling=self.pixdim)
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
            seg_border_dist)) / (np.sum(seg_border + ref_border))
        hausdorff_distance = np.max([np.max(ref_border_dist), np.max(
            seg_border_dist)])
        print(np.max(ref_border_dist))
        print(np.percentile(ref_border_dist[self.seg+self.ref > 0], q=1))

        hd_95 = np.max([np.percentile(ref_border_dist[self.ref+self.seg > 0],
                                      q=95), np.percentile(
            seg_border_dist[self.seg+self.ref > 0], q=95)])
        return hausdorff_distance, average_distance, hd_95

    def header_str(self):
        result_str = [j for i in self.measures for j in self.m_dict[i][1]]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:4f}'):
        result_str = ""
        for i in self.measures:
            for j in self.m_dict[i][0]():
                try:
                    result_str += ',' + fmt.format(j)
                except ValueError:  # some functions give strings e.g., "--"
                    print(i, j)
                    result_str += ',{}'.format(j)
        return result_str
