import os
import random
import  csv
from math import sqrt
from statistics import median, stdev
from sys import argv
from random import seed as inf
import json
import elly_func

sc = elly_func.start_spark('bfr')
inf(553)

class bfr:
    # 把所有收到的點歸屬到DS CS RS
    # Classify nodes into DS, CS and RS.
    def __init__(self, n_cluster, chunk_num):
        self.n_cluster = n_cluster
        self.chunk_num = chunk_num
        self.ds = {}
        self.cs = {}
        self.rs = []


class DSCS:
    def __init__(self, c):
        # cluster code 編號
        self.label = int(c.label)
        # cluster member 內部成員
        self.element = set(c.element.keys())
        # number of cluster members per group 成員總數
        self.element_num = c.element_num
        # cluster dimension 成員 向量維度
        self.dimension = c.dimension
        # Initialization 計算值初始化
        self.sum = self.initial_vector()
        self.sumQ = self.initial_vector()
        self.sigma = self.initial_vector()
        self.cent = self.initial_vector()

        self.initialization(c)
        # Update new centroids 更新質心點
        self.centroid_revise()
    
    # If two elements combined into one, we should recalculate their properties.
    # 若兩點融合 更新計算值
    def combine(self, pt):
        return self.compute_revise(pt.element, pt.sum, pt.sumQ)

    #Set an initial vector to put elements in. 零向量
    def initial_vector(self):
        temp = []
        for i in range(self.dimension):
            v = 0
            temp.append(v)
        return temp

    # Update centroids.
    # 更新質心點
    def centroid_revise(self):
        temp = []
        for i in self.sum:
            k = i / self.element_num
            temp.append(k)
        self.cent = temp

    # Initialization 計算質初始化
    def initialization(self, c):
        for pt in c.element.values():
            for i in range(self.dimension):
                self.sum[i] = self.sum[i] + pt.vec[i]
                self.sumQ[i] = (pt.vec[i]) ** 2 + self.sum[i]
                self.sigma[i] = sqrt(
                    abs(self.sumQ[i] / self.element_num - (self.sum[i] / self.element_num) ** 2))

    # Find if nodes should be classified to CS or DS. 比較歸屬
    def cs_ds(self, a, dic):
        d_min = 1.5e500
        #d_min = float('inf')
        cluster_result = -1
        mah = a * sqrt(self.dimension)
        for i in dic.keys():
            if i != self.label:
                d = self.mah(dic[i].cent, dic[i].sigma)
                if d < d_min:
                    if d <= mah:
                        d_min = abs(d) 
                        cluster_result = i
        return (self.label, cluster_result, d_min)

    # Data 計算值
    def compute_revise(self, pt, sum, sumq):
        self.element_num = self.element_num + len(pt)
        self.element = self.element.union(pt)
        for i in range(self.dimension):
            self.sum[i] = sum[i] + sum[i]
            self.sumQ[i] = sumq[i] + sumq[i]
            self.sigma[i] = sqrt(
                abs(self.sumQ[i] / self.element_num - (self.sum[i] / self.element_num) ** 2))
        self.centroid_revise()
        return self

    # Compute the Mahalanobis distance 馬式距離計算
    def mah(self, cent: list, sigma: list):
        d = 0
        for i in range(self.dimension):
            y = self.cent[i] - cent[i]
            # 如果分母標準差為0，TA說當1計算
            if sigma[i] == 0:
                d = d + y * y
            else:
                d = d + (y / sigma[i]) ** 2
        ans = sqrt(d)
        return ans

# cluster collections 集合
class cluster:

    def __init__(self, label, cent):
        # cluster 編號
        self.label = label
        # cluster 內部成員
        # 初始只有中心點和其個人資料(點編號+座標)
        self.element = {cent.label: cent}
        # 內部成員總數
        self.element_num = len(self.element)
        # 所有點向量維度
        self.dimension = cent.dimension
        self.centroid_revise()

    # New mamber joing to the cluster 新成員加入
    def add_c(self, point):
        self.element[point.label] = point
        self.centroid_revise()

    # Move out the member from the namelist list of the cluster. (cluster 成員驅逐)
    def remove_c(self, point):
        self.element.pop(point.label)
        self.centroid_revise()

    # Update the centroid of the cluster中心點更新
    def centroid_revise(self):
        self.element_num = len(self.element)
        zeros_vec = []
        for i in range(self.dimension):
            zeros_vec.append(0)

        for pt in self.element.values():
            for i in range(pt.dimension):
                zeros_vec[i] = zeros_vec[i] + pt.vec[i]

        temp = []
        for i in zeros_vec:
            c = i / self.element_num
            temp.append(c)
        self.centroid = temp

    # Move out the elements which are too far from its centroid element.
    # 離散點判定 (以兩倍標準差外之點作為離散點) 僅僅淘汰最遠5%的點
    def identify_rs(self):
        b = []
        dic = {}
        for n in self.element.values():
            d = Euclidean_distance(n.vec, self.centroid)
            b.append(d)
            dic[d] = n

        if len(b) > 2:
            med = median(b)
            sigma = stdev(b)

            rs = {}
            for i in dic.keys():
                if i > med + 2 * sigma:
                    self.remove_c(dic[i])
                    rs[dic[i].label] = dic[i]
            return rs



def trans(x):
    data = []
    for i in range(len(x)):
        k = float(x[i])
        data.append(k)
    return data

# Euler distance 歐式距離
def Euclidean_distance(p1: list, p2: list):
    sum1 = 0
    for i in range(len(p1)):
        sum1 = sum1 + (p1[i] - p2[i]) ** 2
    return sqrt(sum1)

# Update personal info. 把所有點map到class建立個人檔案
def point_trans(data):
    x = elly_func.point(data)
    return x

# K-means implementation 實現
def K_means(data: list, k: int, label=0):
    sample = []

    for i in range(len(data)):
        sample.append((i, elly_func.point(data[i])))

    # Randomly pick the first point. 任取第一點
    cluster_initial = 0
    pt1 = random.sample(sample,1)
    # Iterate the rest of centroid elements. 再迭代取剩下的中心點
    first_load = k - 1
    cluster_counter = cluster_initial + 1 +label
    # Collect the centroids. 中心點集合
    seen = [pt1[0][1].vec]
    # Build the dictionary for centroid points. 中心點與其個人資料字典建立
    cluster_dic = {cluster_initial+label: cluster(cluster_initial+label, pt1[0][1])}

    # Find k initial centroids.
    while first_load:
        d_max = -1.5e500
        check = None
    # The point picks the other point that is the farthest from itself. 取距離自己最遠點
        for i in sample:
            d_local = i[1].d_comparison(seen)
            if d_local > d_max:
                check = i[1]
                d_max = d_local
        cluster_dic[cluster_counter] = cluster(cluster_counter, check)
        cluster_counter = cluster_counter + 1
        first_load = first_load - 1
        seen.append(check.vec)

    # Build a library for centroids and their cluster groups. 字典建立: 中心點(長老) 與 他創的cluster(部落)
    cc_dic = {}
    for i, j in cluster_dic.items():
        cc_dic[i] = j.centroid
    counter = 0
    status_revise = 1450

    # Update the centroids. 中心點更新: 若有變更長老，就更新名單，沒有變更長老就跳過此循環
    while status_revise:
        status_revise = 0
        for i in sample:
            i[1].cluster_revise(cc_dic)
            if i[1].status != 0:
                status_revise = status_revise + 1
                cluster_dic[i[1].cluster_result].add_c(i[1])
                if i[1].previous_c != -1:
                    cluster_dic[i[1].previous_c].remove_c(i[1])
        counter = counter + 1

    # Update the centroids. 中心點更新: 長老名單更新
    cc_dic = {}
    for i, j in cluster_dic.items():
        cc_dic[i] = j.centroid
    # Distinguish the elements which are too far from its centroid element.
    # 判定離散點成員有哪些
    rs_dic = {}
    for i, j in cluster_dic.items():
        dic = j.identify_rs()
        if dic is not None:
            rs_dic.update(dic)
        cluster_dic[i] = j
    # If the points which are too far from its centroid element exist,
    # 若有離散點
    if rs_dic:
        rs_list = list(rs_dic.keys()) # Points 離散成員名單
        print('rs_list: ', rs_list)
        cluster_result = -1 # We give the points a label called "-1". 離散成員發給他一張卡 叫-1
        rs_point = random.sample(rs_list, 1)
        print('rs_point: ', rs_point)

        for i in rs_point:
            rs_group = cluster(cluster_result, rs_dic[i])

        for i in rs_dic.keys():
            rs_group.add_c(rs_dic[i])

        cluster_dic[cluster_result] = rs_group
        print('cluster_dic', cluster_dic)
    return cluster_dic

# Initialization 初始化系統
def set_up(sample4k, n_cluster, chunk_num):
    pt = bfr(n_cluster, chunk_num)
    cluster_dic = K_means(sample4k, n_cluster)

    # If there is a points with lable "-1", we put it into RS. 若有離散點(拿到-1卡的人) 丟進RS
    if -1 in cluster_dic.keys():
        for i in cluster_dic[-1].element.values():
            pt.rs.append([i.label] + i.vec)
    # The points without a label "-1" will be put into DS. 沒有拿到(-1) 放DS
    for i in cluster_dic.keys():
        if i != -1:
            pt.ds[i] = DSCS(cluster_dic[i])
    # If The number of the group members is too many, we should keep classifing them into proper groups.
    # 若 RS 離散成員太多 所有部落太多倍 再分類一次
    if len(pt.rs) > 6 * n_cluster:
        cluster_dic_rs = K_means(pt.rs, 4*n_cluster)
        # Update the points we think they should be put in RS. 離散成員名單刷新
        pt.rs.clear()
        # Then put them into RS. 離散成員丟進RS
        for i in cluster_dic_rs[-1].element.values():
            pt.rs.append([i.label] + i.vec)
        # The points we don't take them as the one of RS members at first. 初判定之非離散成員名單
        temp = list(cluster_dic_rs.keys() - [-1])

        # Check again if the point is the memeber in RS or CS. 嚴格審查: 是單槍匹馬才是RS離散人選，若有結伴丟進CS
        for i in temp:
            if cluster_dic_rs[i].element_num < 2:
                for j in cluster_dic_rs[i].element.values():
                    pt.rs.append([j.label] + j.vec)
            else:
                pt.cs[i] = DSCS(cluster_dic_rs[i])
    return pt

# Computing the data. 計算值
def cal(data):
    pt = set()
    temp = []
    for i in range(len(data[1][1][0].vec)):
        a = 0
        temp.append(a)
    sum = temp
    sumQ = temp

    for i in data[1][1]:
        pt.add(i.label)
        sum = list(map(lambda x, y: x + y, sum, i.vec))
        sumQ = list(map(lambda x, y: x + y**2, sumQ, i.vec))
    return (data[0], pt, sum, sumQ)


def main():
    # Upload multipule files. 連續讀好多檔案
    inputPath = r'C:/Users/elysh/PycharmProjects/assignment05/test2/' #argv[1]
    n_cluster = 10 #int(argv[2]) # 部落總數
    output_file1 = '1.json' #argv[3]
    output_file2 = '1.csv' #argv[4]
    chunk = os.listdir(inputPath)
    chunk_num = 0
    Run = inputPath + chunk[chunk_num]
    rddtext = sc.textFile(Run).map(lambda line: line.split(','))
    data = rddtext.map(lambda x: (int(x[0]), trans(x[1:]))).map(
        lambda x: x[1]).collect()
    first_num = len(data)
    print("Total num:", first_num)  # sample    幾%  sample總數
    sam = rddtext.map(trans).sample(False, 0.01, 1450)
    sample4k = sam.collect()
    rdd_sam = sam.map(lambda x: (int(x[0]), point_trans(x)))
    rdd_bfr = rddtext.map(trans).map(lambda x: (int(x[0]), point_trans(x))).subtractByKey(rdd_sam)

    pt = set_up(sample4k, n_cluster, chunk_num)

    row1 = ['round_id', 'nof_cluster_discard', 'nof_point_discard',
                  'nof_cluster_compression', 'nof_point_compression', 'nof_point_retained']

    with open(output_file2,'w') as fs:
        result = csv.writer(fs)
        result.writerow(row1)
    # sqrt(d)倍數係數
    a = 2
    ds_dic = pt.ds
    # The order of the file we are using. 第幾個檔案
    while chunk_num < len(chunk):
        if chunk_num > 0:
            Run = inputPath + chunk[chunk_num]
            rdd_bfr = sc.textFile(Run).map(lambda line: line.split(',')).map(trans).map(
                lambda x: (int(x[0]), point_trans(x)))
            first_num = first_num + rddtext.count()
            print("Total num:", first_num)

        # New member in DS. 新進移民 先分進DS，審查他是否在馬式距離內，是否有資格進部落
        ds_data = rdd_bfr.map(lambda x: x[1].compare_mah(a, ds_dic)).cache()
        # point: (self.cluster_result, self.label,self, self.cs_pre_c)
        # point: (self.cluster_result, self.label,self)
        # If it is the point with the labe "-1". 初判被發(-1)卡的，無法加入部落
        ds = ds_data.filter(lambda line: line[0] != -1).map(lambda x: (x[0], ([x[1]], [x[2]]))).reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])).map(lambda x: cal(x)).collect()
        # compute_revise(self, pt, s, q)
        for i in ds:
                pt.ds[i[0]] = pt.ds[i[0]].compute_revise(i[1], i[2], i[3])
        print(pt.ds.keys())
. 

        # Check the points which are labeled as "-1" and are single 拿-1卡看是否有資格去CS部落，是否有結伴拉伙
        cs_dic = pt.cs
        cs_data = ds_data.filter(lambda line: line[0] == -1).map(lambda x: x[2].compare_mah(a, cs_dic)).cache()
        cs = cs_data.filter(lambda line: line[0] != -1).map(lambda x: (x[0], ([x[1]], [x[2]]))).reduceByKey(
            lambda a, b: (a[0] + b[0], a[1] + b[1])).map(lambda x: cal(x)).collect()
        # point: (self.cluster_result, self.label,self, self.cs_pre_c)
        for i in cs:
                pt.cs[i[0]] = pt.cs[i[0]].compute_revise(i[1], i[2], i[3])
        print(pt.cs.keys())

        # The points are still the ones with a label "-1", and it should be in RS. 最終還是拿(-1)卡的人，真的是單槍匹馬，進RS 離散成員
        rs = cs_data.filter(lambda line: line[0] == -1).map(lambda x: [x[1]] + x[2].vec).collect()

        pt.rs = pt.rs + rs
        print(pt.rs[1])
        print(len(pt.rs))
        #Check if the memebrs in RS is too many.
#       若離散成員太多，再次嚴格審查，拉人進部落
        if len(pt.rs) > 6 * n_cluster:
            cs_label = 0
            if pt.cs != {}:
                cs_list = list(pt.cs.keys())
                cs_label = max(cs_list) + 1
            cluster_dic_rs = K_means(pt.rs, 4*n_cluster, cs_label)
            pt.rs.clear()
            temp = list(cluster_dic_rs.keys())
            if -1 in cluster_dic_rs.keys():
                for i in cluster_dic_rs[-1].element.values():
                    pt.rs.append([i.label] + i.vec)
                temp = list(cluster_dic_rs.keys() - [-1])

            for i in temp:
                if cluster_dic_rs[i].element_num < 2:
                    for j in cluster_dic_rs[i].element.values():
                        pt.rs.append([j.label] + j.vec)
                else:
                    pt.cs[i] = DSCS(cluster_dic_rs[i])
        # Check all. 查看各個部落人數情況，人口普查
        temp = []
        for i in pt.ds.keys():
            k = pt.ds[i].element_num
            temp.append(k)
        s = sum(temp)
        temp = []
        for i in pt.cs.keys():
            k = pt.cs[i].element_num
            temp.append(k)
        w = sum(temp)
        print('Round: ', chunk_num, 'DS: ', s, 'CS: ', w, 'RS: ', len(pt.rs))

        # Check if the points are combined. CS 是否可以融合
        if pt.cs != {}:
            add = 1
            while add == 1:
                temp = []
                for i in pt.cs.keys():
                    temp.append(pt.cs[i].cs_ds(a, cs_dic))
                d_min = min(temp, key=lambda x: x[2])
                if d_min[2] >= 1.5e500:
                    add = 0
                else:
                    pt.cs[d_min[0]] = pt.cs[d_min[0]].combine(pt.cs[d_min[1]])
                    pt.cs.pop(d_min[1])
        # After combing CS, we should classified if it should get in DS. CS融合後 和 DS 判定看是否可以融進DS
        temp = []
        for i in pt.cs.keys():
            cs = pt.cs[i].cs_ds(a, pt.ds)
            if cs[2] < 1.5e500:
                pt.ds[cs[1]] = pt.ds[cs[1]].combine(pt.cs[i])
                temp.append(i)
           
        # If it is DS, we should clear its name in CS. 若進DS 即撤除CS既有名單
        for i in temp:
            del pt.cs[i]
            


        ans_ds = 0
        ans_cs = 0
        ans_cs_num = len(pt.cs)
        ans_rs_num = len(pt.rs)
        ans_total = ans_ds + ans_cs + ans_rs_num

        for i in pt.ds.keys():
            ans_ds = ans_ds + pt.ds[i].element_num
        for i in pt.cs.keys():
            ans_cs = ans_cs + pt.cs[i].element_num
        print('Round: ', chunk_num, 'DS: ', ans_ds, 'CS: ', ans_cs, 'RS: ', ans_rs_num)
        with open(output_file2, 'a') as fs:
            result = csv.writer(fs)
            result.writerow([chunk_num + 1,
                             n_cluster, ans_ds, ans_cs_num, ans_cs, ans_rs_num])
        chunk_num = chunk_num + 1

    temp = {}
    for i in pt.ds.keys():
        for j in pt.ds[i].element:
            temp[str(j)] = i

    rs_label = -1
    for i in pt.cs.keys():
        for j in pt.cs[i].element:
            temp[str(j)] = rs_label

    for i in pt.rs:
        temp[str(i[0])] = rs_label

    with open(output_file1, 'w') as fa:
        json.dump(temp, fa)



if __name__ == '__main__':
    main()
