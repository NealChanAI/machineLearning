"""
1. 定义距离计算公式
2. 随机初始化中心点
3. 迭代划分簇, 更新质点
"""

# 定义距离计算公式
def euclDist(vecA, vecB):
    return np.sqrt(sum(power(vecA-vecB, 2)))

# 随机初始化中心点
def initCentroids(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m))
        centroids[i:] = dataSet[index:]
    return centroids

def KMeans(dataSet, k):
    # 迭代划分簇, 更新质点
    # while True
    clusterSet = np.array(np.zeros((dataSet,shape[0],2)))
    # 初始化中心点
    centroids = initCentroids(dataSet, k)
    clusterChange = True 
    while clusterChange:
        clusterChange = False
        for i in range(dataSet.shape[0]):
            minDist = float("INF")
            minInd = -1
            for j in range(k):
                # 计算距离
                dist = euclDist(dataSet[i: ], centroids[j: ])
                if dist < minDist:
                    minDist = dist
                    minInd = j
                    clusterSet[i, 1] = minDist
            if clusterSet[i, 0] != minInd:
                clusterChange = True
                clusterSet[i, 0] = minInd
                
        # 更新中心点
        for i in range(k):
            cluster = dataSet[np.nonzero(clusterSet[: 0] == i)]
            cetroids[i: ] = np.mean(cluster, axis=0)
    return centroids, clusterSet

----------------------------------------------------
### 网上参考
# 传入数据集和k值
def kmeans(data, k):
    # 计算样本个数
    numSamples = data.shape[0]
    # 样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    clusterData = np.array(np.zeros((numSamples, 2)))
    # 决定质心是否要改变的质量
    clusterChanged = True
    # 初始化质心
    centroids = initCentroids(data, k)
    while clusterChanged:
        clusterChanged = False
        # 循环每一个样本
        for i in range(numSamples):
            # 最小距离
            minDist = 100000.0
            # 定义样本所属的簇
            minIndex = 0
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                # 循环每一个质心和样本，计算距离
                distance = euclDistance(centroids[j, :], data[i, :])
                # 如果计算的距离小于最小距离，则更新最小距离
                if distance < minDist:
                    minDist = distance
                    # 更新最小距离
                    clusterData[i, 1] = minDist
                    # 更新样本所属的簇
                    minIndex = j
            # 如果样本的所属的簇发生了变化
            if clusterData[i, 0] != minIndex:
                # 质心要重新计算
                clusterChanged = True
                # 更新样本的簇
                clusterData[i, 0] = minIndex
        # 更新质心
        for j in range(k):
            # 获取第j个簇所有的样本所在的索引
            cluster_index = np.nonzero(clusterData[:, 0] == j)
            # 第j个簇所有的样本点
            pointsInCluster = data[cluster_index]
            # 计算质心
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
    return centroids, clusterData
