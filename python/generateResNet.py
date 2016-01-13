import caffe
from caffe import layers as L, params as P

def conv_layer(bottom, num_filter, param, weight_filler, bias_filler, kernel = 3, stride = 1, pad = 0):
	conv = L.Convolution(bottom, kernel_size=kernel, stride=stride,
		num_output=num_filter, pad=pad, param = param, weight_filler=weight_filler, bias_filler=bias_filler)	
	return conv

def conv_bn_layers(bottom, num_filter, param, weight_filler, bias_filler, kernel = 3, stride=1, pad = 0):
	conv = conv_layer(bottom, num_filter = num_filter, kernel = kernel, stride=stride, pad = pad,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	#bn = L.BatchNorm(conv, in_place=True, param = [{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])	#use_global_stats = False,
	bn = L.BN(conv, param = [dict(lr_mult=1), dict(lr_mult=1)], scale_filler=dict(type="constant", value=1), shift_filler=dict(type="constant", value=0))
	lrn = L.LRN(bn)
	return lrn

def residual_standard_layers(bottom, param, weight_filler, bias_filler, num_filter):
	conv1 = conv_bn_layers(bottom, num_filter = num_filter, kernel = 3, stride = 1, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	relu1 = L.ReLU(conv1)
	conv2 = conv_bn_layers(relu1, num_filter = num_filter, kernel = 3, stride = 1, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	sum1 = L.Eltwise(conv2, bottom)
	#bn = L.BN(sum1, param = [dict(lr_mult=1), dict(lr_mult=1)], scale_filler=dict(type="constant", value=1), shift_filler=dict(type="constant", value=0))
	#bn = L.BatchNorm(sum1)
	#lrn = L.LRN(bn)
	relu2 = L.ReLU(sum1)
	return relu2

def residual_cluster_layers(bottom, param, weight_filler, bias_filler, num_filter, N):

	res_layers = residual_standard_layers(bottom, num_filter = num_filter,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	current_layers = res_layers
	for i in range(N - 1):
		current_layers = residual_standard_layers(current_layers, num_filter = num_filter,
			param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	return current_layers

def ResCifar(N):
	n = caffe.NetSpec()

	n.data, n.label = L.Data(batch_size=128,
                         backend=P.Data.LMDB, source="lmdb",
                         transform_param=dict(scale=1. / 255), ntop=2)

	param = [dict(lr_mult=1), dict(lr_mult=2)]   
	weight_filler=dict(type="msra")
     	bias_filler=dict(type="constant")

	n.conv1 = conv_bn_layers(n.data, num_filter = 16, kernel = 3, stride = 1, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.res_16 = residual_cluster_layers(n.conv1, num_filter = 16, N = N,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.conv2 = conv_layer(n.res_16, num_filter = 32, kernel = 3, stride = 2, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.res_32 = residual_cluster_layers(n.conv2, num_filter = 32, N = N,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.conv3 = conv_layer(n.res_32, num_filter = 64, kernel = 3, stride = 2, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.res_64 = residual_cluster_layers(n.conv3, num_filter = 64, N = N,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	
	n.global_pool = L.Pooling(n.res_64, pooling_param = dict(pool = P.Pooling.AVE, global_pooling = True))
	n.score = L.InnerProduct(n.global_pool, num_output = 10, 
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	n.loss = L.SoftmaxWithLoss(n.score, n.label)
	n.accuracy = L.Accuracy(n.score, n.label)

	return n

net = ResCifar(5)
text_file = open("Output.txt", "w")
text_file.write(str(net.to_proto()))
text_file.close()
