import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
import os.path


def load_image(path):
	img = tf.io.read_file(path)
	img = tf.io.decode_jpeg(img, channels=3)
	img = tf.image.resize_with_pad(img, 224, 224)
	img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
	return img


def get_image_feature_vectors():
	module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
	module = hub.load(module_handle)
	for filename in glob.glob("./assets/*.jpg"):
		print(filename)
		img = load_image(filename)
		features = module(img)
		feature_set = np.squeeze(features)
		outfile_name = os.path.basename(filename) + ".npz"
		outfile_path = os.path.join("./assets/vectors/", outfile_name)
		np.savetxt(outfile_path, feature_set, delimiter=',')


get_image_feature_vectors()