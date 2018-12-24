import argparse
import sys

import tensorflow as tf
from tensorflow.python.platform import app

updated_node = None


def check_connections(graph_dir):
	graph = tf.GraphDef()
	with tf.gfile.Open(graph_dir, 'r') as f:
		data = f.read()
		graph.ParseFromString(data)

	f = open('{}.txt'.format(graph_dir), 'w')

	for i, node in enumerate(graph.node):
		f.writelines('{} {} {}\n'.format(i, node.name, node.op))
		for j, n in enumerate(node.input):
			f.writelines('  |-------{} {}\n'.format(j, n))
	f.close()


def main(unused_args):
	check_connections(FLAGS.graph_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")
	parser.add_argument(
	  "--graph_dir",
	  type=str,
	  default="",
	  required=True,
	  help="The location of the protobuf (\'pb\') model to visualize.")
	FLAGS, unparsed = parser.parse_known_args()
	app.run(main=main, argv=[sys.argv[0]] + unparsed)