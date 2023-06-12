
from utils.predicter import LabelPredicter



def test_predictor(**kwargs):
    predicter = LabelPredicter(sam_with_label, **kwargs)
    return predicter.debug_predict("validation", data_list_file_path="raw_data/dataset_0.json")

import optuna

def objective(trial):
    #     pred_iou_thresh: float = 0.50,
    # label_certainty_thresh: float = 0.0,
    # #stability_score_thresh: float = 0.95,
    # stability_score_thresh: float = 0.0,
    # stability_score_offset: float = 1.0,
    pred_iou_thresh = trial.suggest_float('pred_iou_thresh', 0.0, 1.0)
    label_certainty_thresh = trial.suggest_float('label_certainty_thresh', 0.0, 1.0)
    stability_score_thresh = trial.suggest_float('stability_score_thresh', 0.0, 1.0)
    stability_score_offset = trial.suggest_float('stability_score_offset', 0.0, 1.0)
    result = test_predictor(
        pred_iou_thresh=pred_iou_thresh,
        label_certainty_thresh=label_certainty_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset
    )
    return result

test_predictor(        pred_iou_thresh=0.80,
        #stability_score_thresh: float = 0.95,
        stability_score_thresh=0,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,)

# # TODO: prune

# study = optuna.create_study(direction='maximize')
# study.enqueue_trial(
#     {
#         "pred_iou_thresh": 0.75,
#         "label_certainty_thresh": 0.3,
#         "stability_score_thresh": 0.3,
#         "stability_score_offset": 1.0
#     }
# )
# study.optimize(objective, n_trials=1000)

# print(study.best_params)