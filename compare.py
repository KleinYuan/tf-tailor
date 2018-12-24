import cv2
import numpy as np
from models.tf_server import TFServer
from utils.utils import load_config
from caffe_classes import class_names

sample = cv2.imread('./example/zebra.jpeg')
sample = cv2.resize(sample, (227, 227))
sample = np.expand_dims(sample, axis=0)


def test_frozen_graph(_config, _data):
	server = TFServer(config=_config)
	server.inference(data=_data)
	class_name = class_names[np.argmax(server.prediction)]
	print("   Predicted class is {}".format(class_name))
	server.clean_up()
	del server


def compare():
	print("Inference Original AlexNet Graph:")
	alexnet_config = load_config('config/alexnet.yaml')
	test_frozen_graph(_config=alexnet_config, _data=[sample])

	print("Inference Tailored AlexNet Graph:")
	alexnet_tailored_config = load_config('config/alexnet_tailored.yaml')
	test_frozen_graph(_config=alexnet_tailored_config,  _data=[sample])


if __name__ == '__main__':
	compare()