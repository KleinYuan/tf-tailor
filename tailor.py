import tensorflow as tf
from tensorflow.core.framework import graph_pb2

RAW_FROZEN_GRAPH_FP = "example/alexnet_frozen.pb"
TRIMMED_FROZEN_GRAPH_FP = "example/alexnet_frozen_tailored.pb"

updated_node = None


# Open Graph
graph = tf.GraphDef()
with tf.gfile.Open(RAW_FROZEN_GRAPH_FP, 'r') as f:
	data = f.read()
	graph.ParseFromString(data)

# Tailor
graph.node[92].input[0] = 'Relu'

nodes = graph.node[:81] + graph.node[88:]
output_graph = graph_pb2.GraphDef()
output_graph.node.extend(nodes)
with tf.gfile.GFile(TRIMMED_FROZEN_GRAPH_FP, 'w') as f:
	f.write(output_graph.SerializeToString())

