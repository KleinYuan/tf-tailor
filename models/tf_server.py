import tensorflow as tf


class TFServer(object):

	def __init__(self, config):
		tf.reset_default_graph()
		self.in_progress = False
		self.prediction = None
		self.session = None
		self.graph = None
		self.frozen = False
		self.feed_dict = {}
		self.output_ops = []
		self.input_ops = []
		self.model_fp = config.model_fp
		self.input_tensor_names = config.input_tensor_names
		self.output_tensor_names = config.output_tensor_names
		self.device = config.device

		with tf.device(self.device):
			self._load_graph()
			self._init_predictor()

	def _load_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.model_fp, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		tf.get_default_graph().finalize()

	def _init_predictor(self):
		tf_config = tf.ConfigProto()
		tf_config.gpu_options.allow_growth = True
		with self.graph.as_default():
			self.session = tf.Session(config=tf_config, graph=self.graph)
			self._fetch_tensors()

	def _fetch_tensors(self):
		assert len(self.input_tensor_names) > 0
		assert len(self.output_tensor_names) > 0
		for _tensor_name in self.input_tensor_names:
			_op = self.graph.get_tensor_by_name(_tensor_name)
			self.input_ops.append(_op)
			self.feed_dict[_op] = None
		for _tensor_name in self.output_tensor_names:
			_op = self.graph.get_tensor_by_name(_tensor_name)
			self.output_ops.append(_op)

	def _set_feed_dict(self, data):
		assert len(data) == len(self.input_ops), 'Data len = {} | Input Ops len = {}'.format(len(data), len(self.input_ops))
		with self.graph.as_default():
			for ind, op in enumerate(self.input_ops):
				self.feed_dict[op] = data[ind]

	def inference(self, data):
		self.in_progress = True

		with self.graph.as_default():
			self._set_feed_dict(data=data)
			self.prediction = self.session.run(self.output_ops, feed_dict=self.feed_dict)
		self.in_progress = False

		return self.prediction

	def get_status(self):
		return self.in_progress

	def clean_up(self):
		# In old version tensorflow
		# session sometimes will not be closed automatically
		self.session.close()
		self.session = None
		tf.reset_default_graph()