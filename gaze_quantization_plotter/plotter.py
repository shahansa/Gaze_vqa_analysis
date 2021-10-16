import os
import json
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .gaze_quantizer import  GazeToObject

plt.rcParams['figure.figsize'] = (8.0, 8.0)

class Plotter:
    """
    plot relevant bounding boxes on top of images and export them to a file
    """
    def __init__(self, args):
        self.save_path = args.save_path
        self.input_fixations = args.input_fixations
        self.vqa_questions_path = args.vqa_questions_path
        self.vqa_images_path = args.vqa_images_path
        self.vqa_image_features_path = args.vqa_image_features_path
        self.fpq_df, self.ques_data = self.build_tables()
        self.question_image_by_id = self.build_question_index()


    def build_tables(self):
        """

        :return:
        """
        fpq_df = pd.read_csv(self.input_fixations)
        with open(self.vqa_questions_path) as fd:
            ques_data = json.load(fd)
        return fpq_df, ques_data

    def build_question_index(self):
        """

        :return:
        """
        ques_data = self.ques_data['questions']
        question_image_by_id = {}
        for entry in ques_data:
            question_image_by_id[entry['question_id']] = (entry['image_id'], entry['question'])
        return question_image_by_id

    def get_image_and_feat_path(self, ques_id):
        """

        :param ques_id:
        :return:
        """
        image_prefix = "COCO_val2014_000000000000"  # general name format of validation set images
        image_id = self.question_image_by_id[ques_id][0]  # replace last n characters of image_prefix with this
        image_prefix = image_prefix[:-(len(str(image_id)))]
        final_img_path = os.path.join(self.vqa_images_path, f"{image_prefix}{image_id}.jpg")
        final_img_feat_path = os.path.join(self.vqa_image_features_path, f"{image_prefix}{image_id}.jpg.npz")
        return final_img_path, final_img_feat_path

    def plot(self, img_path, img_features_path, ques_id, index):
        """

        :param img_path:
        :param img_features_path:
        :param ques_id:
        :param index:
        :return:
        """
        question = self.question_image_by_id[ques_id][1]
        img_feat = np.load(img_features_path)
        bboxes = img_feat['bbox']
        fig, ax = plt.subplots()
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
        gca = ax
        gca.axis('off')
        shape = (img.shape[0], img.shape[1], 1)
        white_wash = np.ones(shape)
        white_wash = white_wash * img
        white_wash = white_wash.astype('uint8')

        quantizer = GazeToObject(bboxes=bboxes, fixations_df=self.fpq_df)
        gaze_to_bbox = quantizer(index=index)

        for bbox, alpha in zip(bboxes, gaze_to_bbox):
            if alpha != 0:
                gca.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                            bbox[2] - bbox[0],
                                            bbox[3] - bbox[1], fill=False, label=alpha,
                                            edgecolor='cyan', linewidth=3))
        gca.imshow(white_wash, interpolation='bicubic')
        gca.axis('off')
        fig.suptitle(question)
        save_path = os.path.join(self.save_path, f"{index}_{ques_id}.png")
        plt.savefig(save_path, dpi=150)

    def __call__(self):
        """

        :return:
        """
        for index in range(self.fpq_df.shape[0]):
            question_ids = self.fpq_df['QuestionId'].tolist()
            ques_id = question_ids[index]
            img_path, img_features_path = self.get_image_and_feat_path(ques_id)
            self.plot(img_path, img_features_path, ques_id, index)



