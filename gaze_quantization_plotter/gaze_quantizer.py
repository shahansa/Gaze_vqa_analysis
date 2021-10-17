import numpy as np
from ast import literal_eval


class GazeToObject:
    """
    Gaze to image object features mapping
    """

    def __init__(self, bboxes, fixations_df, gaze_to_bbox_thresh=0.60):
        self.bboxes = bboxes
        self.fpq_df = fixations_df
        img_orig_x = self.fpq_df['ImageOrigin_x'].tolist()
        img_orig_y = self.fpq_df['ImageOrigin_y'].tolist()
        self.img_orig = [(x, y) for x, y in zip(img_orig_x, img_orig_y)]
        self.image_size_y = self.fpq_df['ImageSize_y'].tolist()
        self.fixations = self.fpq_df['Fixations'].tolist()
        self.gaze_to_bbox_thresh = gaze_to_bbox_thresh

    def __call__(self, index):
        """
        given a question index, find those objects in image to which majority of fixations map
        :param index:
        :return: gaze_to_bbox
        """
        centroid_feat = []
        orig = self.img_orig[index]
        for bbox in self.bboxes:
            centroid = (bbox[0] + ((bbox[2] - bbox[0]) / 2) + orig[0],
                        bbox[1] + ((bbox[3] - bbox[1]) / 2) + (orig[1] - self.image_size_y[index]))
            centroid_feat.append(centroid)
        fix = self.fixations[index]
        fix = literal_eval(fix)
        gaze_to_bbox = np.zeros(len(self.bboxes))
        for fixation in fix:
            distances = []
            for obj in centroid_feat:
                distances.append(np.linalg.norm(np.array((fixation[0], fixation[1])) - np.array(obj)))
            gaze_to_bbox[np.argmin(distances)] += 1
        max_gaze = max(gaze_to_bbox)
        gaze_to_bbox /= max_gaze
        for i in range(len(gaze_to_bbox)):
            if gaze_to_bbox[i] < self.gaze_to_bbox_thresh:
                gaze_to_bbox[i] = 0

        return gaze_to_bbox
