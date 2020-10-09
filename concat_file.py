import cv2
import os


ori_path = '/home/vybt/Downloads/U2_Net_Test'
unet_path = '/home/vybt/Downloads/u-net'
u_2_net_path = '/home/vybt/Downloads/u-2-net-prediction'
final_image_path = '/home/vybt/Downloads/final_image'

ori_image_list = os.listdir(ori_path)
for image_name in ori_image_list:
    try:
        ori_image = cv2.imread(os.path.join(ori_path, image_name))
        mask_name = image_name.replace('.jpg', '.png')

        unet_image = cv2.imread(os.path.join(unet_path, mask_name))
        u_2_net_image = cv2.imread(os.path.join(u_2_net_path, mask_name))

        final_image = cv2.hconcat([ori_image, unet_image, u_2_net_image])
        print(os.path.join(final_image_path, image_name))
        cv2.imwrite(os.path.join(final_image_path, image_name), final_image)
    except Exception as er:
        print(er)
        continue
