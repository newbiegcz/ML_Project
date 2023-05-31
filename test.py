from utils.predicter import LabelPredicter
import numpy as np
from model.build_sam import build_sam_with_label_vit_h
sam_with_label, _ = build_sam_with_label_vit_h()
# print(type(sam_with_label))
predicter = LabelPredicter(sam_with_label)
H=W=30
image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
# print(image.shape)
labels = predicter.predict([image])
print(labels)