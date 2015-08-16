
class DeepNetwork:
    """
    Simple implementation of configurable DNN.
    Number of layers and training data can be specified.

    Layers must be specified using a list-based approach :

        layers = [
            nb of neurons on first layer (in),
            nb of neurons on second layer,
            # ...
            nb of neurons on last layer (out)
        ]

        Synapses will be created accordingly.

    """
    # TODO Allow modification of synapses to reflect network goal development

    def __init__(self, lsizes, in_data, out_data):

        self.in_data = in_data
        self.out_data = out_data

        self.lsizes = lsizes
        self.lnb = len(self.lsizes)

        self._init_synapses()

    def _init_synapses(self):
        self.synapses = []
        for l in range(1,self.lnb):
            self.synapses.append(
                np.random.random((self.lsizes[l-1], self.lsizes[l]))
            )

        self.trained = False


    def NL(self, x, deriv=False):

        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def train(self, condition=lambda _: False, verbose=False):

        if self.trained:
            self._init_synapses()

        self.training_iter = 0

        L = [[] for i in self.lsizes]
        while condition(self):
            if verbose: print('Counter : {}'.format(self.training_iter))

            # forward propagation
            L[0] = self.in_data
            for i in range(1,self.lnb):
                L[i] = self.NL(np.dot(L[i-1], self.synapses[i-1]))

            # TODO backpropagation & synapses regression
            L_err = [0 for i in self.lsizes]
            L_delta = [0 for i in self.lsizes]

            # initialize recur relation on the last element
            L_err[-1] = self.out_data - L[-1]
            L_delta[-1] = L_err[-1]*self.NL(L[-1], True)

            # element-wise backpropagation
            for i in reversed(range(self.lnb-1)):
                L_err[i] = np.dot(L_delta[i+1], self.synapses[i].T)
                L_delta[i] = L_err[i]*self.NL(L[i], True)
                self.synapses[i] += np.dot(L[i].T, L_delta[i+1])

            self.training_iter += 1

        self.trained = True
        return self.synapses

    def train_iter(self, iter_nb):
        return self.train(lambda x: x.training_iter < iter_nb)

    def train_error(self, goal_error, *a, **kw):
        self.train(lambda s: (s.training_iter < 10000) or (np.mean(np.power(np.abs(self.out_data- s.test(self.in_data)), 2)) > goal_error), *a, **kw)

    def test(self, in_data, batch=True):
        if not batch:
            in_data = [in_data]

        L = [[] for i in self.lsizes]
        L[0] = self.in_data
        for i in range(1,self.lnb):
            L[i] = self.NL(np.dot(L[i-1], self.synapses[i-1]))
            
        return L[-1]