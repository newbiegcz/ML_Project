from utils.predicter import LabelPredicter
from modeling.build_sam import sam_with_label_model_registry, build_pretrained_encoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default="checkpoint/extracted.pth")
parser.add_argument('--data_list_file_path', type=str, default="raw_data/dataset_0.json")
parser.add_argument('--file_key', type=str, default="validation")
parser.add_argument('--save_path', type=str, default="result")
parser.add_argument('--device', type=str, default="cuda")

args = parser.parse_args()

print('Loading model...')
sam_with_label, _ = sam_with_label_model_registry['vit_h'](checkpoint=args.checkpoint, build_encoder=False)
# print(sam_with_label.device)
print('Loading encoder...')
sam_with_label.image_encoder = build_pretrained_encoder("vit_h")
sam_with_label.to(args.device)
predicter = LabelPredicter(sam_with_label)
predicter.predict(file_key=args.file_key, data_list_file_path=args.data_list_file_path, save_path=args.save_path)