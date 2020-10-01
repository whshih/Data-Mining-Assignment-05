import json
import csv
from math import sqrt
from statistics import stdev
from statistics import median
import random

def start_spark(name = 'SparkTask', local = 'local[*]'):
    from pyspark import SparkContext
    from pyspark import SparkConf

    confSpark = SparkConf().setAppName(name).setMaster(
        local).set('spark.driver.memory', '4G').set(
            'spark.executor.memory', '4G')
    sc = SparkContext.getOrCreate(confSpark)
    sc.setLogLevel(logLevel='ERROR')
    return sc

def tojson(line):
    lines = line.splitlines()
    data = json.loads(lines[0])
    return data



def Euclidean_distance(p1: list, p2: list):
    sum1 = 0
    for i in range(len(p1)):
        sum1 = sum1 + (p1[i] - p2[i]) ** 2
    return sqrt(sum1)

# 每個成員檔案建立
class point:
    def __init__(self, line):
        self.label = int(line[0])      # 編號
        self.vec = line[1:]            # 向量座標
        self.dimension = len(self.vec) # 座標維度
        self.cluster_result = -1       # 初始每個人都拿(-1)卡
        self.status = 0                # 所屬部落沒動
        self.previous_c = -1           # 移民前部落


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, line):
        vec = [line['label']] + line['vec']
        return self.__init__(vec)

#   馬式距離計算
    def mah(self, centroid: list, sigma: list):
        d = 0
        for i in range(self.dimension):
            w = self.vec[i] - centroid[i]
            if sigma[i] == 0:
                d = d + w*w
            else:
                d = d + (w / sigma[i])**2
        ans = sqrt(d)
        return ans

#   距離比較
    def d_comparison(self, seen: list):
        d_min = 1.5e500
        for i in seen:
            if i == self.vec:
                return -1
            d = Euclidean_distance(i, self.vec)
            if d <= d_min:
                d_min = d
        return d_min

#   cluster更新
    def cluster_revise(self, c_centroid: dict):
        local = 0
        d = 1.5e500
        for i in c_centroid.keys():
            d_local = Euclidean_distance(self.vec, c_centroid[i])
            if d_local < d:
                local = i
                d = d_local
        if  self.cluster_result == -1:
            self.status = 1
            self.cluster_result = local

        elif local != self.cluster_result:
            self.status = 1
            self.previous_c = self.cluster_result
            self.cluster_result = local
        else:
            self.status = 0

#   馬式距離計算比較
    def compare_mah(self, a, pt):
        threshold = a * sqrt(self.dimension)
        for i, j in pt.items():
            d_mah = self.mah(j.cent, j.sigma)
            if d_mah <= threshold:
                self.cluster_result = i
        return (self.cluster_result, self.label, self)



