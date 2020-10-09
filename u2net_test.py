import glob
import os

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils

from data_loader import RescaleT
from data_loader import SalObjDataset
from data_loader import ToTensorLab
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

FULL_BODY_CLASSES = {
    "background": (0, 0, 0), #0
    "top": (35, 35, 125), #1
    "bottom": (255, 0, 255), #2
    "shoes": (125, 35, 35), #3
    "accessories": (70, 70, 70), #4
    "skin_right_arm": (0, 0, 255), #5
    "skin_left_arm": (0, 255, 0), #6
    "skin_right_leg": (85, 255, 170), #7
    "skin_left_leg": (0, 255, 255), #8
    "hair":  (35, 125, 200), #9
    "skin_face_neck": (255, 255, 0), #10
}

FULL_BODY_LABEL_COLORS = np.array([FULL_BODY_CLASSES['background'],
                                    FULL_BODY_CLASSES['top'],
                                    FULL_BODY_CLASSES['bottom'],
                                    FULL_BODY_CLASSES['shoes'],
                                    FULL_BODY_CLASSES['accessories'],
                                    FULL_BODY_CLASSES['skin_right_arm'],
                                    FULL_BODY_CLASSES['skin_left_arm'],
                                    FULL_BODY_CLASSES['skin_right_leg'],
                                    FULL_BODY_CLASSES['skin_left_leg'],
                                    FULL_BODY_CLASSES['hair'],
                                    FULL_BODY_CLASSES['skin_face_neck']],
                                   dtype=np.float32)


BPS_RGB_COLORS = {
    "background": (0, 0, 0),
    "head": (100, 0, 255),
    "neck": (155, 0, 0),
    "trunk": (0, 0, 155),
    "left_arm": (255, 255, 0),
    "right_arm": (0, 255, 255),
    "left_leg": (170, 255, 0),
    "right_leg": (255, 0, 255),
}


LIST_BPS_COLORS = [color[::-1] for color in BPS_RGB_COLORS.values()]
LIST_BGR_COLORS = [color[::-1] for color in FULL_BODY_CLASSES.values()]

# Switch to list of BGR colors


def classes_matrix_to_color_image(pred_mask, width, height):
    output = np.zeros(shape=(height, width, 3),
                      dtype='uint8')
    for i in range(8):
        indices = np.where(pred_mask == i)
        output[indices] = LIST_BPS_COLORS[i]

    return output


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

import cv2
def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    # print(predict_np.shape)
    predict_np = np.rollaxis(predict_np, 0, 3)
    # print(predict_np.shape)
    image = io.imread(image_name)
    prediction = cv2.resize(predict_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    prediction_argmax = np.argmax(prediction, axis=2)
    outputs = np.uint8(prediction_argmax)

    final = classes_matrix_to_color_image(outputs, image.shape[1], image.shape[0])
    img_name = image_name.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    cv2.imwrite(os.path.join(d_dir, imidx + '.png'), final)
    # imo.save(os.path.join(d_dir, imidx + '.png'))
#

def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2net' #u2netp

    image_dir = "/home/vybt/Downloads/U2_Net_Test"
    prediction_dir = "/home/vybt/Downloads/u-2--bps-net-prediction"
    model_dir = '/media/vybt/DATA/SmartFashion/deep-learning-projects/U-2-Net/saved_models/u2net/_bps_bce_itr_300000_train_0.107041_tar_0.011690.pth'

    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 8)
    elif model_name == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 8)

    net.load_state_dict(torch.load(model_dir))

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("Inference: ", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # print("inputs test: {}".format(inputs_test.shape))
        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()
