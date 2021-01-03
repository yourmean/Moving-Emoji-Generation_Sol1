import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def img_file_to_gif(img_files, output_file_name):
    ## imge 파일 리스트로부터 gif 생성 
    imgs_array = [np.array(imageio.imread(img_file)) for img_file in img_file_lst]

    imageio.mimsave(output_file_name, imgs_array, duration=0.5)
