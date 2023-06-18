import numpy as np
import ml_project.utils.visualize as visualize
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default="img0035")
parser.add_argument('--save_path', type=str, default="result")
args = parser.parse_args()

file_name = args.file_name
result_dice_file = os.path.join(args.save_path, f'{file_name}_dice.npy')
result_dice = np.load(result_dice_file)

img_file = os.path.join(args.save_path, f'{file_name}_imgs.npy')
img = np.load(img_file)

pd_label_file = os.path.join(args.save_path, f'{file_name}_pd_labels.npy')
pd_label = np.load(pd_label_file)

gt_label_file = os.path.join(args.save_path, f'{file_name}_gt_labels.npy')
gt_label = np.load(gt_label_file)

print(result_dice.shape) # (13,)
print(pd_label.shape, gt_label.shape, img.shape) # (?, 1024, 1024)

img = img.transpose(3, 0, 1, 2)
print(img.shape)

visualize.initialize_window()
visualize.add_object_3d("image3D",
                    image=img,
                    pd_label=pd_label,
                    gt_label=gt_label,
                    prompt_points=None,
                    label_name=visualize.default_label_names,
                    extras=None
)