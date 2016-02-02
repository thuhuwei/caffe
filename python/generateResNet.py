import sys   
sys.setrecursionlimit(1000000)

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
	relu1 = L.ReLU(conv1, in_place=True)
	conv2 = conv_bn_layers(relu1, num_filter = num_filter, kernel = 3, stride = 1, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	sum1 = L.Eltwise(conv2, bottom)
	relu2 = L.ReLU(sum1, in_place=True)
	return relu2

def residual_bottle_layers(bottom, param, weight_filler, bias_filler, num_filter, stage_no, cluster_no):
	proj = bottom
	stride = 1

	if cluster_no == 0:
		if stage_no > 0:
			stride = 2

		proj = conv_layer(bottom, num_filter = num_filter * 4, kernel = 1, stride = stride, pad = 0,
			param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	conv1 = conv_bn_layers(bottom, num_filter = num_filter, kernel = 1, stride = stride, pad = 0,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	relu1 = L.ReLU(conv1, in_place=True)

	conv2 = conv_bn_layers(relu1, num_filter = num_filter, kernel = 3, stride = 1, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	relu2 = L.ReLU(conv2, in_place=True)

	conv3 = conv_bn_layers(relu2, num_filter = num_filter * 4, kernel = 1, stride = 1, pad = 0,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	sum1 = L.Eltwise(conv3, proj)
	relu3 = L.ReLU(sum1, in_place=True)
	return relu3

def residual_standard_cluster_layers(bottom, param, weight_filler, bias_filler, num_filter, N):

	res_layers = residual_standard_layers(bottom, num_filter = num_filter,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	current_layers = res_layers
	for i in range(N - 1):
		current_layers = residual_standard_layers(current_layers, num_filter = num_filter,
			param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	return current_layers

def residual_bottle_cluster_layers(bottom, param, weight_filler, bias_filler, num_filter, N, stage_no):

	res_layers = residual_bottle_layers(bottom, num_filter = num_filter,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler, stage_no = stage_no, cluster_no = 0)
	
	current_layers = res_layers
	for i in range(N - 1):
		current_layers = residual_bottle_layers(current_layers, num_filter = num_filter,
			param = param, weight_filler = weight_filler, bias_filler = bias_filler, stage_no = stage_no, cluster_no = i + 1)

	return current_layers

def ResImageNet(total_depth):
	n = caffe.NetSpec()

	n.data, n.label = L.Data(batch_size=128,
                         backend=P.Data.LMDB, source="lmdb",
                         transform_param=dict(scale=1. / 255), ntop=2)

	param = [dict(lr_mult=1), dict(lr_mult=2)]   
	weight_filler=dict(type="msra")
     	bias_filler=dict(type="constant")

	
	net_defs = {
		18:([2, 2, 2, 2], "standard"),
		34:([3, 4, 6, 3], "standard"),
		50:([3, 4, 6, 3], "bottleneck"),
		101:([3, 4, 23, 3], "bottleneck"),
		152:([3, 8, 36, 3], "bottleneck"),
	    }
	nunits_list, unit_type = net_defs[total_depth]
	num_filters = [64, 128, 256, 512]

	n.conv1 = conv_bn_layers(n.data, num_filter = 64, kernel = 7, stride = 2, pad = 3,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)	
	n.relu1 = L.ReLU(n.conv1, in_place=True)
	n.pool1 = L.Pooling(n.relu1, stride = 2, kernel_size = 3)	
	current_layer = n.pool1

	for idx in range(4):
		num_filter = num_filters[idx]
		nunits = nunits_list[idx]

		if idx > 0:
			if unit_type == "standard":
				current_layer = conv_layer(current_layer, num_filter = num_filter, kernel = 3, stride = 2, pad = 1,
					param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	
		if unit_type == "standard":
			current_layer = residual_standard_cluster_layers(current_layer, num_filter = num_filter, N = nunits,
				param = param, weight_filler = weight_filler, bias_filler = bias_filler)
		else:
			current_layer = residual_bottle_cluster_layers(current_layer, num_filter = num_filter, N = nunits,
				param = param, weight_filler = weight_filler, bias_filler = bias_filler, stage_no = idx)

	n.global_pool = L.Pooling(current_layer, pooling_param = dict(pool = P.Pooling.AVE, global_pooling = True))
	n.score = L.InnerProduct(n.global_pool, num_output = 1000, 
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	n.loss = L.SoftmaxWithLoss(n.score, n.label)
	n.accuracy = L.Accuracy(n.score, n.label)

	return n

def ResCifar10(N):
	n = caffe.NetSpec()

	n.data, n.label = L.Data(batch_size=128,
                         backend=P.Data.LMDB, source="lmdb",
                         transform_param=dict(scale=1. / 255), ntop=2)

	param = [dict(lr_mult=1), dict(lr_mult=2)]   
	weight_filler=dict(type="msra")
     	bias_filler=dict(type="constant")

	n.conv1 = conv_bn_layers(n.data, num_filter = 16, kernel = 3, stride = 1, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.res_16 = residual_standard_cluster_layers(n.conv1, num_filter = 16, N = N,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.conv2 = conv_layer(n.res_16, num_filter = 32, kernel = 3, stride = 2, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.res_32 = residual_standard_cluster_layers(n.conv2, num_filter = 32, N = N,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.conv3 = conv_layer(n.res_32, num_filter = 64, kernel = 3, stride = 2, pad = 1,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)

	n.res_64 = residual_standard_cluster_layers(n.conv3, num_filter = 64, N = N,
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	
	n.global_pool = L.Pooling(n.res_64, pooling_param = dict(pool = P.Pooling.AVE, global_pooling = True))
	n.score = L.InnerProduct(n.global_pool, num_output = 10, 
		param = param, weight_filler = weight_filler, bias_filler = bias_filler)
	n.loss = L.SoftmaxWithLoss(n.score, n.label)
	n.accuracy = L.Accuracy(n.score, n.label)

	return n

#net = ResCifar10(7)
net = ResImageNet(34)
text_file = open("train_val.prototxt", "w")
text_file.write(str(net.to_proto()))
text_file.close()
