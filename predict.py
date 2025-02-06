import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from models.unets import Unet2D
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from models.FCN import FCN_Vgg16_16s

from utils.learning.metrics import dice_coef, precision, recall, IoU
from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.data import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGen



def predict(base_path, pred_name, weight_file_name,input_dim=(512,512), color_space='rgb'):
    """Predicts on base_path + test/images,
    stores predictions under base_path + test/images/pred_name/
    uses model under training_history/weight_file_name"""
    input_dim_x, input_dim_y = input_dim
    data_gen = DataGen(base_path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
    x_test, test_label_filenames_list = load_test_images(base_path)
    pred_filename_list =[]
    for name in test_label_filenames_list:
        pred_filename_list.append(name[:-4] + '-prediction.png')

    # ### get unet model
    # unet2d = Unet2D(n_filters=64, input_dim_x=input_dim_x, input_dim_y=input_dim_y, num_channels=3)
    # model = unet2d.get_unet_model_yuanqing()
    # model = load_model('./azh_wound_care_center_diabetic_foot_training_history/' + weight_file_name
    #                , custom_objects={'recall':recall,
    #                                  'precision':precision,
    #                                  'dice_coef': dice_coef,
    #                                  'relu6':relu6,
    #                                  'DepthwiseConv2D':DepthwiseConv2D,
    #                                  'BilinearUpsampling':BilinearUpsampling})

    # ### get separable unet model
    # sep_unet = Separable_Unet2D(n_filters=64, input_dim_x=input_dim_x, input_dim_y=input_dim_y, num_channels=3)
    # model, model_name = sep_unet.get_sep_unet_v2()
    # model = load_model('./azh_wound_care_center_diabetic_foot_training_history/' + weight_file_name
    #                , custom_objects={'dice_coef': dice_coef,
    #                                  'relu6':relu6,
    #                                  'DepthwiseConv2D':DepthwiseConv2D,
    #                                  'BilinearUpsampling':BilinearUpsampling})

    # ### get VGG16 model
    # model, model_name = FCN_Vgg16_16s(input_shape=(input_dim_x, input_dim_y, 3))
    # with CustomObjectScope({'BilinearUpSampling2D':BilinearUpSampling2D}):
    #     model = load_model('./azh_wound_care_center_diabetic_foot_training_history/' + weight_file_name
    #                    , custom_objects={'dice_coef': dice_coef})

    # ### get mobilenetv2 model
    model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
    model = load_model('./training_history/' + weight_file_name
                   , custom_objects={'recall':recall,
                                     'precision':precision,
                                     'dice_coef': dice_coef,
                                     'IoU': IoU,
                                     'relu6':relu6,
                                     'DepthwiseConv2D':DepthwiseConv2D,
                                     'BilinearUpsampling':BilinearUpsampling})

    for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
        prediction = model.predict(image_batch, verbose=1)
        try:
            os.makedirs(base_path + 'test/predictions/' + pred_name + '/', exist_ok=False)
        except FileExistsError:
            print(f"Directory {base_path + 'test/predictions/' + pred_name + '/'} already exists, creating a second one!")
            pred_name += '(1)'
            os.makedirs(base_path + 'test/predictions/' + pred_name + '/', exist_ok=False)
        save_results(prediction, 'rgb', base_path + 'test/predictions/' + pred_name + '/', pred_filename_list)
        break


if __name__ == "__main__":
    input_dim_x = 512
    input_dim_y = 512
    color_space = 'rgb'
    base_path = './data/fuseg_small/'
    weight_file_name = '2025-02-05 10:22:01.366389.hdf5'
    pred_name = '2025-02-05-366389-short_train-14eps/'
    predict(base_path, pred_name, weight_file_name, input_dim=(input_dim_x, input_dim_y), color_space=color_space)