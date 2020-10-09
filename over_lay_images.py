import cv2
import os

images_path = '/home/vybt/Downloads/U2_Net_Test'
label_path = '/home/vybt/Downloads/u-2--bps-net-prediction'

list_original_images = os.listdir(images_path)

for image_name in list_original_images:

    image = cv2.imread(os.path.join(images_path, image_name))
    label = cv2.imread(os.path.join(label_path, image_name.replace('.jpg', '.png')))
    output = label.copy()
    cv2.addWeighted(image, 0.5, output, 0.5, 0, output)

    cv2.imwrite(os.path.join(label_path, image_name), output)