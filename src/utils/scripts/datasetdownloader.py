### Adaptation of this original code: https://github.com/DataSciencePolimi/Visual-TCAV/blob/main/datasets_and_models_downloader/datasets_and_models_downloader.py
### Downloads a set of images from ImageNet for a set of classes, and saves them in the ACG4CAV data folder. The number of images per class can be set by the user, as well as the option to download random images or not. The code also extracts one image per class and saves it in the ACG4CAV data folder with the class name as filename, to be used as single image for testing.

# Imports
if __name__ == "__main__":
	# import os
	import random
	import requests
	import json
	import time
	import numpy as np
	import multiprocessing

headers = {
	"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:95.0) Gecko/20100101 Firefox/95.0",
}

# Parameters
random_download = False
n_images_per_class = 50
n_single_images_per_class = 2
n_random_images = 50
imagenet_classes = [
	"n02391049",  # Zebra
]

# Paths
from os import path, listdir, makedirs, remove

acg4cav_path = r"../../../"

acg4cav_data_path = path.join(acg4cav_path, "data")
acg4cav_images_path = path.join(acg4cav_data_path, "images")

imagenet_class_info_file = path.join(acg4cav_data_path, "imagenet_class_info.json")

if __name__ == "__main__":
	print("Creating folders...", end=' ')

	makedirs(acg4cav_images_path, exist_ok=True)

	print("Done!")

# Images download
url_to_scrape = lambda wnid: f'http://www.image-net.org/api/imagenet.synset.geturls?wnid={wnid}'

if __name__ == "__main__":
	class_info_dict = dict()
	with open(imagenet_class_info_file) as class_info_json_f:
		class_info_dict = json.load(class_info_json_f)


def get_image_name(url):
	name = url.split('/')[-1].split("?")[0]
	return name


def get_image(url):
	if not url.lower().endswith(".jpg") and not url.lower().endswith(".jpeg"):
		return False
	try:
		import requests
		img_resp = requests.get(url, timeout=(3, 3), headers=headers)
	except:
		return False
	if not 'content-type' in img_resp.headers:
		return False
	if not 'image' in img_resp.headers['content-type']:
		return False
	if (len(img_resp.content) < 1000):
		return False
	img_name = get_image_name(url)
	if (len(img_name) <= 1):
		return False
	return img_resp


def get_image_from_url(url, img_file_path, n):
	img_resp = get_image(url)
	if not img_resp:
		return False
	img_name = str(n) + ".jpg"
	from os import path
	with open(path.join(img_file_path, img_name), 'wb') as img_f:
		img_f.write(img_resp.content)
	return True


# Test images
if __name__ == "__main__":

	print("Downloading test images... this may take a while...", end=' ')

	for imagenet_class in imagenet_classes:

		response = requests.get(url_to_scrape(imagenet_class), headers=headers)
		try:
			urls_to_scrape = np.array([url.decode('utf-8') for url in response.content.splitlines()])
		except:
			break
		img_file_path = path.join(acg4cav_images_path, class_info_dict[imagenet_class]["class_name"])
		makedirs(img_file_path, exist_ok=True)
		if random_download:
			random.shuffle(urls_to_scrape)

		procs = dict()
		for i, url in enumerate(urls_to_scrape):

			rem = min(n_images_per_class, len(urls_to_scrape)) - len(listdir(img_file_path))
			if rem <= 0:
				break

			if len(procs) < min(rem, 100):
				procs[i] = multiprocessing.Process(target=get_image_from_url, args=(url, img_file_path, i,))
				procs[i].start()
			else:
				for proc in list(procs):
					procs[proc].join()
					procs.pop(proc)

			time.sleep(0.1)

		for proc in list(procs):
			procs[proc].join()
			procs.pop(proc)

		for i, file in enumerate(listdir(img_file_path)):
			if i >= n_images_per_class:
				remove(path.join(img_file_path, file))

		# Extract one image per class
		to_export = random.choice([dir for dir in listdir(img_file_path) if dir.endswith(".jpg")])
		import shutil

		shutil.copyfile(path.join(img_file_path, file), path.join(acg4cav_images_path,
																  class_info_dict[imagenet_class][
																	  "class_name"] + ".jpg"))

	for i in range(n_random_images):
		random_class = random.choice(imagenet_classes)
		response = requests.get(url_to_scrape(random_class), headers=headers)
		try:
			urls_to_scrape = np.array([url.decode('utf-8') for url in response.content.splitlines()])
		except:
			break
		random_url = random.choice(urls_to_scrape)
		get_image_from_url(random_url, acg4cav_images_path, "negatives")
