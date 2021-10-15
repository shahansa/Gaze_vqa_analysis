import argparse
import os

home = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description='Gaze quantization into objects and plotting',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--save_path', type=str, default=os.path.join(home, 'plots'), help="Path to store plots")
parser.add_argument('--input_fixations', type=str,
                    default=os.path.join(home, 'data/vqa_v2_gaze_1_650_all.csv'),
                    help="path to input csv which has fixations per question data")
parser.add_argument('--vqa_questions_path', type=str,
                    default=os.path.join(home, 'data/v2_OpenEnded_mscoco_val2014_questions.json'),
                    help="VQA V2 input questions")
parser.add_argument('--vqa_images_path', type=str, default=os.path.join(home, 'data/val2014'),
                    help="VQA input images path")
parser.add_argument('--vqa_image_features_path', type=str,
                    default=os.path.join(home, 'data/features_val2014'),
                    help="VQA input images features path. for feature extraction, "
                         "refer to https://github.com/yuzcccc/bottom-up-attention")

if __name__ == "__main__":
    args = parser.parse_args()
    plotter = Plotter(args)
    try:
        plotter()
    except Exception as e:
        print(e.message)
        exit(-1)
