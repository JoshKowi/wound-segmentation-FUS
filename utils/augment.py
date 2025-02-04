import Augmentor

p = Augmentor.Pipeline("./data/Foot_Ulcer_Segmentation_Challenge/train/images")
p.ground_truth("./data/Foot_Ulcer_Segmentation_Challenge/train/labels")
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.set_save_format(save_format='auto')
p.sample(10000, multi_threaded=True)
