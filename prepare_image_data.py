import os 
import numpy as np
from skimage import io 
from PIL import Image

def get_image_path(imgs_dir):
    all_images =os.listdir(imgs_dir)
    all_images_paths=[]
    try:
        for image in all_images:
            image_path=os.path.join(imgs_dir,image)
            for i in os.listdir(image_path):
                if i.endswith('png'):
                    full_image_path=os.path.join(image_path,i)
                    all_images_paths.append(full_image_path)
                    # print(all_images_paths)
    except NotADirectoryError:
        pass
    return all_images_paths



import cv2

def get_min_height(images):
    all_heights=[]
    for image in images:
        im = cv2.imread(image)
        height=im.shape[0]
        all_heights.append(height)
    min_height = min(all_heights)
    print(min_height)
    return min_height


def resized_img_dir():
    final_destination ='/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/resized_images'
    if not os.path.isdir(final_destination):
        os.mkdir(final_destination)
    return final_destination

def resize_images(image_path, save_dir):
    new_height=get_min_height(images=get_image_path(imgs_dir='/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/images'))
    for i, image in enumerate(image_path):
        im=cv2.imread(image,cv2.IMREAD_COLOR)
        if im.shape[-1]==3:
            try:
                width = im.shape[1]
                height =new_height
                dim=(width,height)
                resized_image=cv2.resize(im,dim,interpolation=cv2.INTER_AREA)
                # Save the resized image to the save_dir directory
                cv2.imwrite(f'{save_dir}/image_{i}.jpg', resized_image)
                cv2.imshow('image',resized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                pass

if __name__=='__main__':
    get_image_path(imgs_dir='/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/images'
)
get_min_height(images=get_image_path(imgs_dir='/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/images'))

resized_img_dir()

resize_images(image_path=get_image_path(imgs_dir='/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/images'),
              save_dir=resized_img_dir())
