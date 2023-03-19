
# Defing a class for each layer in the network
# contain T_start, T_previsous, T_compute, T_done, D_compute, D_communicate
class Layer:
    def __init__(self, name):
        self.name = name
        self.T_start = None
        self.T_previous = None
        self.T_compute = None
        self.T_done = None
        self.D_compute = None
        self.D_communicate = None

embedding = Layer('embedding')
ls_encoder = Layer('ls_encoder')
evaluate = Layer('evaluate')
fake = Layer('fake')

# 4 GPU
# evaluate.D_communicate = 4.698
# ls_encoder.D_communicate = 12.2
# embedding.D_communicate = 27.477
# fake.D_communicate = 0

# evaluate.D_compute = 0.483
# ls_encoder.D_compute = 6.94
# embedding.D_compute = 2.22
# fake.D_compute = 0
# overhead = 0.09

# 2 GPU
'''
embedding & 1.8 & 22.45 \\
ls\_encoder & 7.4 & 9.0 \\
evaluate & 0.513 & 1.773 \\
'''
evaluate.D_communicate = 1.773
ls_encoder.D_communicate = 9.0
embedding.D_communicate = 22.45
fake.D_communicate = 0

evaluate.D_compute = 0.513
ls_encoder.D_compute = 7.4
embedding.D_compute = 1.8
fake.D_compute = 0
overhead = 0.09

layer_list = []
# 1 evaluate + 12 * encoder + 1 embedding + 1 fake
layer_list.append(evaluate)
for i in range(12):
    layer_list.append(ls_encoder)
layer_list.append(embedding)
layer_list.append(fake)

layer_list[0].T_start = 0
layer_list[0].T_previous = 0
n_layers = len(layer_list) - 1
print("Total layers:", n_layers)
for i in range(len(layer_list)):
    if i == n_layers:
        fake_layer = layer_list[-1]
        print("Complete time:", fake_layer.T_previous)
        break
    layer_list[i].T_compute = layer_list[i].T_start + layer_list[i].D_compute
    layer_list[i+1].T_start = layer_list[i].T_compute + overhead
    if layer_list[i].T_previous > layer_list[i].T_compute:
        layer_list[i].T_done = layer_list[i].T_previous + layer_list[i].D_communicate
    else:
        layer_list[i].T_done = layer_list[i].T_compute + layer_list[i].D_communicate
    layer_list[i+1].T_previous = layer_list[i].T_done

    
