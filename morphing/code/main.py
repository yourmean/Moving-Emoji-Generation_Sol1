from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_sequence
from image_to_gif import img_file_to_gif

import subprocess
import argparse
import shutil
import os
import cv2

import json


def doMorphing(img1, landmark2, num, save_image_path):

	points2 = landmark2
	[size, img1, points1, list3] = generate_face_correspondences(img1)
	tri = make_delaunay(size[1], size[0], list3)

	generate_morph_sequence(img1, points1, points2, tri, size, save_image_path, num)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--img1" ,required= True, help="The First Image")
	parser.add_argument("--landmark_path" ,required= True, help="landmark dir")
	parser.add_argument("--save_image_path" ,required= True, help="save_image dir")
	parser.add_argument("--output", default='results/output.gif',help="Output Video Path")
	args = parser.parse_args()

	image1 = cv2.imread(args.img1)
	
	landmark_dir = args.landmark_path 
	file_list = os.listdir(landmark_dir)

	# input landmark 형식에 따라 바꿔야 함
	for i in range(len(file_list)):

		with open(file_list[i], 'r') as f:
			json_data = json.load(f)
		landmark2 = json_data
		doMorphing(image1, landmark2, num, args.save_image_path)

	
img_list = os.listdir((args.save_image_path)
img_file_to_gif(img_list, args.output)

print('complete')

#### sample image delete 
for img_file in img_list:
    if os.path.exists(img_file):
        os.remove(img_file)
print('image file delete complete')
