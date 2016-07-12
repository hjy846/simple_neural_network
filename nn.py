#! -*- encoding:utf-8 -*-
import sys
import numpy as np
import pickle

class MyNode:
    def __init__(self, i, layer):
        self.a = 0
        self.i = i
        self.layer = layer

    def __unicode__(self):
        return "%s\t%s\t%s" % (self.a, self.i, self.layer)

    def print_str(self):
        print "%s\t%s\t%s" % (self.a, self.i, self.layer)

class MyNN:
    def __init__(self):
        print '__init__'
        self.initStruct()

    #初始化网络, layer_node表示网络结构， [4,5,1]表示输入层为4个节点，隐藏层有5个节点，1为输出层，目前仅测试了一个输出层的结果
    def initStruct(self, layer_node = [4,5, 1]):
        self.weights = []
        self.bias = []

        # init weight
        layer = 1
        last_layer_num = layer_node[0]
        for nd in layer_node[1:]:
            #随机给weight赋值，均值为0， 方差为0.1
            weight_normal = np.random.normal(0, 0.1, (nd, last_layer_num))
            self.weights.append(weight_normal)
            last_layer_num = nd
            bias_normal = np.random.normal(0, 0.1, nd).reshape(nd, 1)
            #print 'bias_normal: '
            #print bias_normal
            self.bias.append(bias_normal)

    def saveNN(self, weight_path, bias_path):
        pickle.dump(self.weights, open(weight_path, 'w'))
        pickle.dump(self.bias, open(bias_path, 'w'))

    def loadNN(self, weight_path, bias_path):
        self.weights = pickle.load(open(weight_path))
        self.bias = pickle.load(open(bias_path))
            
    def loadTrainData(self):
        pass

    def train(self, train_data, study_rate = 1, l2 = 0.1):
        layer_num = len(self.weights)
        for count in range(1000):
            print 'count:%s' % count
            a = [None] * layer_num
            delta = [None] * layer_num
            delta_w = []
            delta_b = []
            for w in self.weights:
                delta_w.append(np.zeros(w.shape))

            for b in self.bias:
                delta_b.append(np.zeros(b.shape))
                #print delta_w, delta_b
            data_num = len(train_data.T)
            for data in train_data.T:
                len_data = len(data)
                features = data[0:len_data - 1]
                features = features.reshape(len_data - 1,1)
                label = data[len_data - 1:len_data]
                layer_matrix = features
                #print layer_matrix
                layer = 0
                for w in self.weights:
                    z_matrix = w.dot(layer_matrix) + self.bias[layer]
                    layer_matrix = self.activeFunc(z_matrix)
                    a[layer] = layer_matrix
                    layer += 1
                #print 'a'
                #print a

                f_zL = a[layer_num - 1] * (1 - a[layer_num - 1])
                #print f_zL
                delta[layer_num - 1] = (a[layer_num - 1] - label) * f_zL

                for i in range(layer_num - 1)[::-1]:
                    f_zl = a[i] * (1 - a[i])
                    delta[i] = (self.weights[i + 1].T.dot(delta[i + 1])) * f_zl
                #print delta

                round_w = []
                #first_round_w = features.dot(delta[0].T)
                first_round_w = delta[0].dot(features.T)
                round_w.append(first_round_w)
                for i in range(layer_num - 1):
                    #r_w = a[i].dot(delta[i + 1].T)
                    r_w = delta[i + 1].dot(a[i].T)
                    round_w.append(r_w)

                round_b = delta
                #print 'round w, b'
                #print round_w, round_b
                

                for i in range(layer_num):
                    delta_w[i] += round_w[i]
                    delta_b[i] += round_b[i]

                #print 'delta_w'
                #print delta_w
                #print 'delta_b'
                #print delta_b
            for i in range(layer_num):
                self.weights[i] = self.weights[i] - study_rate * ((1.0 / data_num) * delta_w[i])
                #self.weights[i] = self.weights[i] - study_rate * ((1.0 / data_num) * delta_w[i] + l2 * self.weights[i])
                self.bias[i] = self.bias[i] - study_rate * ((1.0 / data_num) * delta_b[i])

        print 'final weights, bias'
        print self.weights, self.bias                
        print 'end final weights, bias'

    def activeFunc(self, value):
        return 1 / (1 + np.exp(-value))

    def predictOne(self, predict_data):
        layer_matrix = predict_data
        layer = 0
        for w in self.weights:
            z_matrix = layer_matrix.dot(w) + self.bias[layer]
            layer_matrix = self.activeFunc(z_matrix)
            layer += 1

        #print layer_matrix
        return layer_matrix

    def predict(self, test_data):
        layer_matrix = test_data
        layer = 0
        print self.weights
        print self.bias
        for w in self.weights:
            z_matrix = w.dot(layer_matrix) + self.bias[layer]
            layer_matrix = self.activeFunc(z_matrix)
            layer += 1

        return layer_matrix

def load_train_data(path):
    ret = []
    with open(path) as fp:
        for line in fp:
            item = line.strip().split(',')
            if len(item) < 5:continue
            if item[4] == 'Iris-setosa':
                item[4] = 0
            else: item[4] = 1
            item = [float(x) for x in item]
            ret.append(item)
    return np.array(ret)

def genTrainData():
    ret = []
    for i in range(10000):
        label = i % 2
        item = np.array([label])
        item2 = np.random.rand(3)
        item3 = np.hstack((item, item2))
        item4 = np.hstack((item3, item))
        ret.append(item4)
    return np.array(ret)

if __name__ == '__main__':
    nn = MyNN()
    #nn.saveNN('weights.file', 'bias.file')
    #nn.loadNN('weights.file', 'bias.file')
    """
    for node in nn.nodes:
        for i in node:
            i.print_str()

    for weight_layer in nn.weights:
        print weight_layer
    """
    all_data = load_train_data('iris.data')
    all_data = genTrainData()
    print all_data
    random_index = np.ones(len(all_data), dtype=bool) 
    random_index[np.random.choice(range(len(all_data)), int(len(all_data) * 0.2), replace=False)] = False
    train_data = all_data[random_index]
    #print train_data
    test_data = all_data[~random_index]
    #print len(all_data), len(train_data), len(test_data)
    nn.train(train_data.T)
    #nn.train(np.array([[1,1,1,1,1]]).T)
    #nn.saveNN('weights.file', 'bias.file')
    test_result = test_data[:,4:5].T
    predict_result = nn.predict(test_data[:,0:4].T)

    test_result = np.array(test_result.flat).astype(int)
    #print test_result
    #print predict_result
    final_predict_result = np.array([1 if x >= 0.5 else 0 for x in predict_result.flat])
    #print final_predict_result
    
    result = (~np.logical_xor(test_result,final_predict_result)).astype(int)
    print np.mean(result)
    
