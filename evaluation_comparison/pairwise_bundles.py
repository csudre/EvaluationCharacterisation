from __future__ import absolute_import, print_function

import numpy as np
from scipy import ndimage
from functools import partial
from scipy.spatial.distance import cdist
from scipy.stats import entropy

class CharBundle(object):
    def __init__(self, streamlines, affine, shape):
        self.streamlines = streamlines
        self.shape = shape
        self.affine = affine
        self.lengths = self.get_lengths()
        self.offsets = self.get_offsets()
        self.number = self.get_number()
        self.indices = self.get_indices()
        self.start = self.get_start()
        self.end = self.get_end()
        self.prob_map = self.make_prob_map()
        self.indices_full = self.create_indices_full()


    def get_indices(self):
        track_values = self.streamlines.streamlines.data
        ind_mm_1exp = np.concatenate((track_values, np.ones(
            [track_values.shape[0], 1])), -1)
        inv_affine = np.linalg.inv(self.affine)
        ind_array = np.asarray(np.matmul(inv_affine, ind_mm_1exp.T),
                               dtype=int).T
        return ind_array

    def get_offsets(self):
        return self.streamlines.streamlines._offsets

    def get_lengths(self):
        return self.streamlines.streamlines._lengths

    def get_number(self):
        return len(self.streamlines.streamlines._lengths)

    def get_start(self):
        list_start = []
        for f in self.offsets:
            list_start.append(self.indices[f,:])
        return np.asarray(list_start)

    def get_end(self):
        list_end = []
        for f in self.offsets:
            if f>0:
                list_end.append(self.indices[f-1,:])
        list_end.append(self.indices[-1,:])
        return np.asarray(list_end)

    def make_prob_map(self):
        prob_map = np.zeros(self.shape)
        for ind in self.indices:
            prob_map[ind[0],ind[1],ind[2]] +=1
        prob_map *= 1.0/self.number
        return prob_map

    def create_indices_full(self):
        prob_indices = np.asarray(np.where(self.prob_map > 0)).T
        full_indices = prob_indices[:, 0] + prob_indices[:, 1] * self.shape[0] \
                       + prob_indices[:, 2] * self.shape[0] * self.shape[1]
        return full_indices

    def create_stream_id(self):
        list_id = []
        for s in range(0, self.number):
            list_id.append(np.ones([self.lengths[s]])*s)
        return np.concatenate(list_id, 0)

    def create_extent_mapstream(self):
        full_indices = self.indices_full
        numb_indices = len(full_indices)
        list_indices = self.indices[:, 0] + self.indices[:,1]*self.shape[0] +\
                       self.indices[:,2] * self.shape[0]*self.shape[1]
        numb_streams = self.number
        stream_id = self.create_stream_id()
        stream_id = np.asarray(stream_id, dtype=int)
        map_streams = np.zeros([numb_indices, numb_streams])
        for ind in range(0, numb_indices):
            fi = full_indices[ind]
            find_fi = np.asarray(np.where(list_indices == fi))
            if np.where(list_indices==fi)[0].shape[0] > 0:
                find_fi = np.reshape(find_fi, [-1])
                for f in find_fi:
                    #print(f)
                    map_streams[ind, stream_id[f]] = 1
        return map_streams





class PairwiseMeasuresStreams(object):
    def __init__(self, seg_streams, ref_streams,
                 measures=None, num_neighbors=8, pixdim=[1, 1, 1],
                 empty=False, list_labels=None):

        self.m_dict = {
            'ref volume': (self.vol_ref, 'Volume_(Ref)'),
            'seg volume': (self.vol_seg, 'Volume_(Seg)'),
            'ref numb': (self.numb_ref, 'Numb_(Ref-bg)'),
            'seg numb': (self.numb_seg, 'Numb_(Seg-bg)'),
            'dist_com': (self.dist_com, 'DistCom'),
            'start_comp': (self.start_comp, ['StartComp', 'NumbRemainSSeg',
                                             'NumbRemainSRef', 'DistSSeg',
                                             'DistSRef']),
            'end_comp': (self.end_comp, ['EndComp', 'NumbRemainESeg',
                                             'NumbRemainERef', 'DistESeg',
                                             'DistERef']),
            'prob_match': (self.prob_comp, ['DiceDiff','DiceAtch']),
            'length_match': (self.length_match, 'LengthMatch'),
            'summary_match': (self.summary_matching, ['DivMatching',
                              'MinMatching','MaxMatching', 'MeanMatching'])
            #'streamlines_averagedist': (self.stream_dist, 'StreamDist'),

        }
        self.seg = seg_streams
        self.ref = ref_streams
        self.flag_empty = empty
        self.measures = measures if measures is not None else self.m_dict
        self.m_dict_result = {}
        self.pixdim = pixdim

    def vol_ref(self):
        return np.sum(self.ref.prob_map)

    def vol_seg(self):
        return np.sum(self.seg.prob_map)

    def numb_ref(self):
        return self.ref.number

    def numb_seg(self):
        return self.seg.number

    def dist_com(self):
        print("performing distance com")
        prob_seg = self.seg.prob_map
        prob_ref = self.ref.prob_map
        ind_seg = np.asarray(np.where(prob_seg>0)).T
        weight_seg = [prob_seg[ind_seg[i, 0], ind_seg[i, 1], ind_seg[i,
                                                                     2]] for
                      i in range(0, ind_seg.shape[0])]
        com_seg = np.sum(ind_seg * np.tile(np.reshape(weight_seg, [-1,1]),
                                           [1,3] ), 0) / np.sum(weight_seg)
        ind_ref = np.asarray(np.where(prob_ref>0)).T
        weight_ref = [prob_ref[ind_ref[i, 0], ind_ref[i, 1], ind_ref[i,
                                                                     2]] for
                      i in range(0, ind_ref.shape[0])]
        com_ref = np.sum(ind_ref * np.tile(np.reshape(weight_ref, [-1, 1]),
                                           [1, 3]), 0)/np.sum(weight_ref)
        return np.sqrt(np.sum(np.square(com_ref - com_seg)))



    def length_match(self):
        print("Performing length matching")
        min_length = np.min([np.min(self.ref.lengths), np.min(
            self.seg.lengths)])
        max_length = np.max([np.max(self.ref.lengths), np.max(
            self.seg.lengths)])
        distribution_ref = np.zeros([max_length-min_length+1])
        distribution_seg = np.zeros([max_length-min_length+1])
        for s in self.seg.lengths:
            distribution_seg[s-min_length] +=1
        for r in self.ref.lengths:
            distribution_ref[r-min_length] +=1
        normalised_distref = distribution_ref * 1.0 / np.sum(distribution_ref)
        normalised_distseg = distribution_seg * 1.0 / np.sum(distribution_seg)
        print(np.sum(normalised_distref), np.sum(normalised_distseg))

        kld = entropy(normalised_distref+0.000001, qk=normalised_distseg+0.000001) + \
              entropy(
            normalised_distseg+0.000001, qk=normalised_distref+0.000001)
        return kld * 0.5

    def prob_comp(self):
        print("Performing comparison prob map")
        prob_seg = self.seg.prob_map
        prob_ref = self.ref.prob_map

        prob_seg_back = 1 - prob_seg
        prob_ref_back = 1 - prob_ref

        geom_seg = np.sqrt(prob_seg * prob_seg_back)
        geom_ref = np.sqrt(prob_ref * prob_ref_back)

        dist_atch_fore = np.square(np.log((prob_seg + 0.0000001) / (geom_seg+
                                                            0.0000001))\
                         - np.log((prob_ref + 0.0000001) / (geom_ref+
                                                            0.0000001)))
        dist_atch_back = np.square(np.log((prob_seg_back + 0.0000001) / (
                geom_seg +
                                                                   0.0000001)) \
                                   - np.log((prob_ref_back + 0.0000001) / (
                                               geom_ref + 0.0000001)))
        dist_atch = np.sqrt(dist_atch_fore + dist_atch_back)
        final_f = 1.0/(1+dist_atch)
        final_f = np.where(prob_seg+prob_ref>0, final_f, np.zeros_like(final_f))
        dice_atch = np.sum(final_f)/np.count_nonzero(prob_seg+prob_ref)

        diff = np.abs(prob_seg - prob_ref)
        diff_back = np.abs(prob_seg_back - prob_ref_back)
        map_diff = 1 - 0.5 * (diff + diff_back)
        map_common = np.where(prob_ref+prob_seg > 0, np.ones_like(prob_seg),
                              np.zeros_like(prob_seg))
        dice_diff = np.sum(map_diff * map_common) / np.count_nonzero(prob_seg +
                                                             prob_ref)

        return dice_diff, dice_atch


        #
        # comp = np.sum(prob_seg * prob_ref) / np.sum(prob_seg + prob_ref)
        # return comp

    def start_comp(self):
        print("Performing start comparison")
        start_seg = self.seg.start
        start_ref = self.ref.start
        numb_common_start, list_seg_remain, list_ref_remain = \
            self.compare_list_indices(start_seg, start_ref)
        if np.asarray(list_ref_remain).shape[0] > 0 and np.asarray(
                list_seg_remain).shape[0] > 0:
            distremain_seg, distremain_ref = self.mean_dist_mismatch(
                list_seg_remain, list_ref_remain)
        else:
            distremain_ref = 0
            distremain_seg = 0
        numb_remainseg = np.asarray(list_seg_remain).shape[0]
        numb_remainref = np.asarray(list_ref_remain).shape[0]
        return numb_common_start, numb_remainseg, numb_remainref, \
               distremain_seg, distremain_ref

    def end_comp(self):
        print("performing end comparison")
        end_seg = self.seg.end
        end_ref = self.ref.end
        numb_common_start, list_seg_remain, list_ref_remain = \
            self.compare_list_indices(end_seg, end_ref)
        if np.asarray(list_ref_remain).shape[0] > 0 and np.asarray(
                list_seg_remain).shape[0] > 0:
            distremain_seg, distremain_ref = self.mean_dist_mismatch(
                list_seg_remain, list_ref_remain)
        else:
            distremain_ref = 0
            distremain_seg = 0
        numb_remainseg = np.asarray(list_seg_remain).shape[0]
        numb_remainref = np.asarray(list_ref_remain).shape[0]
        return numb_common_start, numb_remainseg, numb_remainref, \
               distremain_seg, distremain_ref


    def find_best_matching_stream(self, ref_mapstream, stream_seg_ind):
        #print("Finding best matching")
        full_ind_ref = self.ref.indices_full
        list_full_ind_ref = list(full_ind_ref)
        stream_full_ind = self.transform_ind_full(stream_seg_ind,
                                                  self.seg.shape)
        list_mapping = []

        for ind in range(0, len(stream_full_ind)):
            if stream_full_ind[ind] in full_ind_ref:
                ind_found = list_full_ind_ref.index(stream_full_ind[ind])
                list_mapping.append(np.expand_dims(ref_mapstream[ind_found,
                                                   :],0))
        if len(list_mapping) > 0:
            array_mapping = np.concatenate(list_mapping, 0)
            #print(array_mapping.shape)

            matching_sum = np.sum(array_mapping, 0)
            return np.argmax(matching_sum), np.max(matching_sum)
        else:
            return 0, 0

    def summary_matching(self):
        print("Performing summary matching")
        array_summary = np.zeros([self.seg.number, 3])
        map_ref = self.ref.create_extent_mapstream()
        print("map created")
        for s in range(0, self.seg.number):
            #print("Streamline %d" %s)
            stream_seg_ind = self.seg.indices[self.seg.offsets[s] : self.seg.offsets[s] + self.seg.lengths[s], :]
            matching_ref, numb_matching = self.find_best_matching_stream(
                map_ref,  stream_seg_ind)
            array_summary[s, 0] = matching_ref
            array_summary[s, 1] = numb_matching
            array_summary[s, 2] = self.seg.lengths[s]
        ratio_match = array_summary[:, 1] * 1.0 / array_summary[: ,2]
        return len(np.unique(array_summary[:, 0])), np.min(ratio_match), \
               np.max(ratio_match), np.mean(ratio_match)



    @staticmethod
    def transform_ind_full(list_ind, shape):
        full_ind = list_ind[:, 0] + list_ind[:,1] * shape[0] + list_ind[:,
                                                              2] * shape[0] *\
                   shape[1]
        return full_ind


    @staticmethod
    def compare_list_indices(start_seg, start_ref):
        list_seg_matched = []
        list_ref_matched = []
        list_seg_remained = []
        list_ref_remained = []
        for s in range(0, start_seg.shape[0]):
            ind_seg = start_seg[s, :]
            for r in range(0, start_ref.shape[0]):
                if r not in list_ref_matched:
                    ind_ref = start_ref[r, :]
                    if np.sum(np.abs(ind_ref-ind_seg)) == 0:
                        list_seg_matched.append(s)
                        list_ref_matched.append(r)
                        continue
            if s not in list_seg_matched:
                list_seg_remained.append(np.expand_dims(ind_seg,0))
        numb_matched = len(list_seg_matched)
        for r in range(0, start_ref.shape[0]):
            ind_ref = start_ref[r, :]
            if r not in list_ref_matched:
                list_ref_remained.append(np.expand_dims(ind_ref,0))
        if len(list_ref_remained) > 0:
            concat_ref = np.concatenate(list_ref_remained, 0)
        else:
            concat_ref = []
        if len(list_seg_remained) > 0:
            concat_seg = np.concatenate(list_seg_remained, 0)
        else:
            concat_seg = []
        return numb_matched, concat_seg, concat_ref

    @staticmethod
    def mean_dist_mismatch(mismatch_seg, mismatch_ref):
        if mismatch_seg.shape[0] == 0 or mismatch_ref.shape[0] ==0:
            return 0
        cdist_segref = cdist(mismatch_seg, mismatch_ref)
        min_distref = np.mean(np.min(cdist_segref, 0))
        min_distseg = np.mean(np.min(cdist_segref, 1))
        return min_distseg, min_distref

    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str_fin = []
        for strres in result_str:
            if isinstance(strres, list):
                new_strres = ','.join(strres)
                result_str_fin.append(new_strres)
            else:
                result_str_fin.append(strres)

        result_str_fin = ',' + ','.join(result_str_fin)
        return result_str_fin

    def to_string(self, fmt='{:4f}'):
        result_str = ""
        for i in self.measures:
            # print(i, self.m_dict[i])
            test = self.m_dict[i][0]()
            if isinstance(test, (tuple, list, np.ndarray)):
                for j in test:
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
            else:
                j = test
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

    # def to_string(self, fmt='{:.4f}'):
    #     result_str = ""
    #     list_space = ['com_ref', 'com_seg', 'list_labels']
    #     for key in self.measures:
    #         result = self.m_dict[key][0]()
    #         if key in list_space:
    #             result_str += ' '.join(fmt.format(x) for x in result) \
    #                 if isinstance(result, tuple) else fmt.format(result)
    #         else:
    #             result_str += ','.join(fmt.format(x) for x in result) \
    #                 if isinstance(result, tuple) else fmt.format(result)
    #         result_str += ','
    #     return result_str[:-1]  # trim the last comma


