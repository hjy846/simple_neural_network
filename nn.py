#! -*- encoding:utf-8 -*-
import sys
import numpy as np
import pickle

class MyNN:
    def __init__(self):
        self.initStruct()

    """
    #初始化网络, layer_node表示网络结构， [4,5,1]表示输入层为4个节点，隐藏层有5个节点，1为输出层，目前仅测试了一个输出层的结果
    #weight和bias随机赋值，随机值服从正态分布，方差为sigma
    """
    def initStruct(self, layer_node = [4, 5, 1], sigma = 0.1):
        self.weights = []
        self.bias = []

        # init weight
        layer = 1
        last_layer_num = layer_node[0]
        for nd in layer_node[1:]:
            #随机给weight赋值，均值为0， 方差为sigma
            weight_normal = np.random.normal(0, sigma, (nd, last_layer_num))
            self.weights.append(weight_normal)
            last_layer_num = nd
            bias_normal = np.random.normal(0, sigma, nd).reshape(nd, 1)
            #print 'bias_normal: '
            #print bias_normal
            self.bias.append(bias_normal)

    """
        保存weights和bias结果，方便直接加载数据，不用再训练
    """
    def saveNN(self, weight_path, bias_path):
        pickle.dump(self.weights, open(weight_path, 'w'))
        pickle.dump(self.bias, open(bias_path, 'w'))

    """
        加载weights和bias数据
    """
    def loadNN(self, weight_path, bias_path):
        self.weights = pickle.load(open(weight_path))
        self.bias = pickle.load(open(bias_path))
            
    """
        训练网络，total_step是迭代次数， batch_size是随机梯度下降算法的batch大小，study_rate是学习率， l2是正则项，加了这个参数后不知为啥拟合得不好，暂时没用
    """
    def train(self, train_data, total_step = 10000, batch_size = 100, study_rate = 0.1, l2 = 0.1):
        layer_num = len(self.weights)
        for step in range(total_step):
            print 'step:%s' % step
            a = [None] * layer_num
            delta = [None] * layer_num
            delta_w = []
            delta_b = []
            for w in self.weights:
                delta_w.append(np.zeros(w.shape))

            for b in self.bias:
                delta_b.append(np.zeros(b.shape))

            #随机抽取batch_size样本
            indexs = np.arange(len(train_data.T))
            indexs = np.random.choice(indexs, batch_size, replace = False)
            batch_data = (train_data.T)[indexs]
            for data in batch_data:
                len_data = len(data)
                features = data[0:len_data - 1]
                features = features.reshape(len_data - 1,1)
                label = data[len_data - 1:len_data]
                layer_matrix = features

                #前向计算
                layer = 0
                for w in self.weights:
                    z_matrix = w.dot(layer_matrix) + self.bias[layer]
                    layer_matrix = self.activeFunc(z_matrix)
                    a[layer] = layer_matrix
                    layer += 1

                #反向传播
                f_zL = a[layer_num - 1] * (1 - a[layer_num - 1])
                delta[layer_num - 1] = (a[layer_num - 1] - label) * f_zL
                for i in range(layer_num - 1)[::-1]:
                    f_zl = a[i] * (1 - a[i])
                    delta[i] = (self.weights[i + 1].T.dot(delta[i + 1])) * f_zl

                #计算weight和bias的梯度
                round_w = []
                first_round_w = delta[0].dot(features.T)
                round_w.append(first_round_w)
                for i in range(layer_num - 1):
                    r_w = delta[i + 1].dot(a[i].T)
                    round_w.append(r_w)

                round_b = delta

                for i in range(layer_num):
                    delta_w[i] += round_w[i]
                    delta_b[i] += round_b[i]

            #计算完batch总体梯度后，更新weights和bias
            for i in range(layer_num):
                self.weights[i] = self.weights[i] - study_rate * ((1.0 / batch_size) * delta_w[i])
                #self.weights[i] = self.weights[i] - study_rate * ((1.0 / batch_size) * delta_w[i] + l2 * self.weights[i])
                self.bias[i] = self.bias[i] - study_rate * ((1.0 / batch_size) * delta_b[i])

        #print 'final weights, bias'
        print self.weights, self.bias                
        #print 'end final weights, bias'

    """
        激活函数，sigmoid
    """
    def activeFunc(self, value):
        return 1 / (1 + np.exp(-value))

    """
        预测接口
    """
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

"""
    加载iris数据集
"""
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

"""
    生成随机训练数据，[y, x, x, x, y], 前四个是特征值，最后一个是label， 只与第一个特征正相关
"""
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

    #all_data = load_train_data('iris.data')
    all_data = genTrainData()
    #分割数据， 80%的为训练集， 20%为测试集
    random_index = np.ones(len(all_data), dtype=bool) 
    random_index[np.random.choice(range(len(all_data)), int(len(all_data) * 0.2), replace=False)] = False
    train_data = all_data[random_index]
    test_data = all_data[~random_index]

    #开始训练
    nn.train(train_data.T)
    
    #测试集结果
    test_result = test_data[:,4:5].T
    
    #预测结果, 为概率值
    predict_result = nn.predict(test_data[:,0:4].T)
    test_result = np.array(test_result.flat).astype(int)
    #转化为相应的标签：0 or 1
    final_predict_result = np.array([1 if x >= 0.5 else 0 for x in predict_result.flat])
    
    #验证准确率
    result = (~np.logical_xor(test_result,final_predict_result)).astype(int)
    print np.mean(result)
    
