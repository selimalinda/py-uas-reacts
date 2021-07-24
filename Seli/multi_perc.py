import numpy as np
class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layer =[self.num_inputs]+self.num_hidden+[self.num_outputs]

        self.weights =[]
        for i in range(len(layer)-1):
            w = np.random.rand(layer[1], layer[1+1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        activation = inputs

        for w in self.weights:
            #calculate net
            net_input = np.dot(activation,w)

            activation = self._sigmoid(net_input)

        return activation

    def _sigmoid(self, x):
        return 1/(1+ np.exp(-x))

if __name__=="_main_":
    mlp = MLP()
    #create some input
    inputs = np.random.rand(mlp.num_inputs)
    #perform
    outputs = mlp.forward_propagate(inputs)
    #print the result
    print("The Network input is : {}".format(inputs))
    print("The Network input is : {}".format(inputs))