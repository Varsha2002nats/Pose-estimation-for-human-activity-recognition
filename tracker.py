import numpy as np
import functools

class Tracker(object):
    ''' A simple tracker for skeletons:

        For previous skeletons (S1) and current skeletons (S2),
        S1[i] and S2[j] are matched, if:
        1. S2[j] is the nearest skeleton to S1[i].
        2. S1[i] is the nearest skeleton to S2[j].
        3. The distance between S1[i] and S2[j] is smaller than `self._dist_thresh`.
        
        Unmatched skeletons in S2 are considered new people.
    '''

    def __init__(self, dist_thresh=0.4, max_humans=5):
        ''' 
        Arguments:
            dist_thresh {float}: Maximum distance to match skeletons. 
                (Image width and height are normalized to [0, 1].)
            max_humans {int}: Maximum number of humans to track.
        '''
        self._dist_thresh = dist_thresh
        self._max_humans = max_humans
        self._dict_id2skeleton = {}
        self._cnt_humans = 0

    def track(self, curr_skels):
        ''' Track the input skeletons by matching them with previous skeletons.
        
        Arguments:
            curr_skels {list of list}: Each sub-list represents a skeleton (keypoints).

        Returns:
            self._dict_id2skeleton {dict}: A dictionary mapping human ID to their skeleton.
        '''
        curr_skels = self._sort_skeletons_by_dist_to_center(curr_skels)
        N = len(curr_skels)

        # Match skeletons between previous and current frames
        if len(self._dict_id2skeleton) > 0:
            ids, prev_skels = map(list, zip(*self._dict_id2skeleton.items()))
            good_matches = self._match_features(prev_skels, curr_skels)

            self._dict_id2skeleton = {}
            is_matched = [False] * N
            for i2, i1 in good_matches.items():
                human_id = ids[i1]
                self._dict_id2skeleton[human_id] = np.array(curr_skels[i2])
                is_matched[i2] = True

            unmatched_idx = [i for i, matched in enumerate(is_matched) if not matched]
        else:
            unmatched_idx = range(N)

        # Add unmatched skeletons as new humans
        num_humans_to_add = min(len(unmatched_idx), self._max_humans - len(self._dict_id2skeleton))
        for i in range(num_humans_to_add):
            self._cnt_humans += 1
            self._dict_id2skeleton[self._cnt_humans] = np.array(curr_skels[unmatched_idx[i]])

        return self._dict_id2skeleton

    def _get_neck(self, skeleton):
        ''' Get the neck keypoint (x, y). '''
        return skeleton[2], skeleton[3]

    def _sort_skeletons_by_dist_to_center(self, skeletons):
        ''' Sort skeletons by their distance to the image center (0.5, 0.5). '''
        def calc_dist(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        def cost(skeleton):
            x, y = self._get_neck(skeleton)
            return calc_dist((x, y), (0.5, 0.5))

        return sorted(skeletons, key=functools.cmp_to_key(lambda a, b: (cost(a) > cost(b)) - (cost(a) < cost(b))))

    def _match_features(self, features1, features2):
        ''' Match skeletons between features1 and features2. '''
        features1, features2 = np.array(features1), np.array(features2)

        def calc_dist(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        def cost(sk1, sk2):
            # Consider joints for matching
            joints = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 25])
            sk1, sk2 = sk1[joints], sk2[joints]
            valid_idx = np.logical_and(sk1 != 0, sk2 != 0)
            sk1, sk2 = sk1[valid_idx], sk2[valid_idx]

            if len(sk1) == 0:
                return float('inf')  # No valid points
            sum_dist = np.mean([calc_dist(sk1[i:i+2], sk2[i:i+2]) for i in range(0, len(sk1), 2)])
            return sum_dist

        # Create distance matrix
        dist_matrix = np.array([[cost(f1, f2) for f2 in features2] for f1 in features1])
        good_matches = {}

        # Find matches
        if len(features1) and len(features2):
            matches_f1_to_f2 = dist_matrix.argmin(axis=1)
            matches_f2_to_f1 = dist_matrix.argmin(axis=0)
            for i1, i2 in enumerate(matches_f1_to_f2):
                if matches_f2_to_f1[i2] == i1 and dist_matrix[i1, i2] < self._dist_thresh:
                    good_matches[i2] = i1

        return good_matches
