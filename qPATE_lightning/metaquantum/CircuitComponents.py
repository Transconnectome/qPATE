import pennylane as qml

class VariationalQuantumBlock:
	def __init__(
			self,
			in_channels=10,
			out_channels=4,
			num_of_qubits=10,
			featuremap_depth=2,
			variational_depth=2,
			hadamard_gate = True,
			entanglement = 'linear',
			qdevice = "lightning.gpu"):

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_of_qubits = num_of_qubits
		self.variational_depth = variational_depth
		self.featuremap_depth = featuremap_depth
		self.hadamard_gate = hadamard_gate

		# parsing entangler map 
		if entanglement == 'linear': 
			# entangling each qubits with its neighbor
			self.entangler_map = [[i, i + 1] for i in range(self.num_of_qubits - 1)]
			self.entangler_map.append([self.num_of_qubits-1,0])
		elif entanglement == 'alternative_linear': 
			self.entangler_map = []
			for i in range(0, self.num_of_qubits - 1, 2):
				self.entangler_map.append([i, i + 1])
			for i in range(0, self.num_of_qubits - 2, 2):
				self.entangler_map.append([i + 1, i + 2])		
		
		# set device
		self.qdevice = qdevice
		self.dev = qml.device(self.qdevice, wires = num_of_qubits, batch_obs = True)

		# Initialize quantum circuit
		self.qlayer = self.circuit()


	def FeatureEmbedding(self, inputs):
		"""
		(ref) qml.AngleEmbedding: https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html
		Angle Embedding -> Haramard
        """
		qml.AngleEmbedding(features=inputs, wires=range(self.num_of_qubits), rotation='Z')
		if self.hadamard_gate == True:
			for i in range(self.num_of_qubits): 
					qml.Hadamard(wires=i)
	
	
	def VariationalLayers(self, weights): 
		"""
		ref 1: https://pennylane.ai/qml/demos/tutorial_variational_classifier
		ref 2: https://github.com/Transconnectome/qPATE_GAN/blob/QML-GPUs/QML-GPUS/run_qnn_pytorch.py#L244
		"""
		# Rotation
		for j in range(self.num_of_qubits): 
			qml.Rot(weights[j,0], weights[j,1], weights[j,2], wires=j)
			# Entangling 
			for src, target in self.entangler_map: 
				qml.CNOT(wires=[src,target])
	

	def circuit(self): 
		weight_shapes = {"weights": (self.variational_depth, self.num_of_qubits, 3)}
		@qml.qnode(self.dev, diff_method='adjoint')
		def _circuit(inputs, weights): 
			for i in range(self.featuremap_depth):
				self.FeatureEmbedding(inputs=inputs)
			for j in range(self.variational_depth):
				self.VariationalLayers(weights=weights[j])
			return [qml.expval(qml.PauliZ(k)) for k in range(self.out_channels)]
		return qml.qnn.TorchLayer(_circuit, weight_shapes)
		

	def _forward(self, inputs):
		"""The variational classifier."""
		raw_output = self.qlayer(inputs)
		return raw_output

	def forward(self, inputs):
		fw = self._forward(inputs)
		return fw
