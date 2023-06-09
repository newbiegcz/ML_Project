from modeling.build_sam import sam_with_label_model_registry, build_pretrained_encoder
from utils.predicter import LabelPredicter

sam_with_label, _ = sam_with_label_model_registry['vit_h'](checkpoint="extracted.pth", build_encoder=False)
del sam_with_label.image_encoder
sam_with_label.add_module("image_encoder", build_pretrained_encoder("vit_h"))
sam_with_label.to("cuda")

predicter = LabelPredicter(sam_with_label)
predicter.predict("validation", data_list_file_path="/root/autodl-tmp/raw_data/dataset_0.json")

