# GBDT二分类

GBDT无论是做分类任务还是回归任务都是使用**CART回归树**作为基分类器。

二分类任务使用与逻辑回归相同的对数似然损失作为损失函数，对强分类器进行偏导（强分类器是多个若分类器的累加和）。

## 逻辑回归的对数损失函数

![image-20210719091329592](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091329592.png)

函数 ![[公式]](https://www.zhihu.com/equation?tex=h_%7B%5Ctheta%7D%28x%29) 的值有特殊的含义，它表示结果取 ![[公式]](https://www.zhihu.com/equation?tex=1) 的概率，因此对于输入 ![[公式]](https://www.zhihu.com/equation?tex=x) 分类结果为类别 ![[公式]](https://www.zhihu.com/equation?tex=1) 和类别 ![[公式]](https://www.zhihu.com/equation?tex=0) 的概率分别为：

![image-20210719091347730](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091347730.png)

下面我们根据上式，推导出逻辑回归的对数损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%29) 。上式综合起来可以写成：

![image-20210719091423844](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091423844.png)

然后取似然函数为：

![image-20210719091438130](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091438130.png)

因为 ![[公式]](https://www.zhihu.com/equation?tex=l%28%5Ctheta%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=logl%28%5Ctheta%29) 在同一 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 处取得极值，因此我们接着取对数似然函数为：

![image-20210719091451484](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091451484.png)

最大似然估计就是求使![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%29) 取最大值时的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 。这里对 ![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta%29) 取相反数，可以使用梯度下降法求解，求得的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 就是要求的最佳参数：

![image-20210719091506639](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091506639.png)

## GBDT二分类原理

逻辑回归单个样本 ![[公式]](https://www.zhihu.com/equation?tex=%28x_%7Bi%7D%2Cy_%7Bi%7D%29) 的损失函数可以表达为：

![image-20210719091532344](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091532344.png)

其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_%7Bi%7D%7D+%3D+h_%7B%5Ctheta%7D%28x%29) 是逻辑回归预测的结果。假设GBDT第 ![[公式]](https://www.zhihu.com/equation?tex=M) 步迭代之后当前学习器为 ![[公式]](https://www.zhihu.com/equation?tex=F%28x%29+%3D+%5Csum_%7Bm%3D0%7D%5EM+h_m%28x%29) ，将 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D) 替换为 ![[公式]](https://www.zhihu.com/equation?tex=F%28x%29)带入上式之后，可将损失函数写为：![image-20210719091601068](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091601068.png)

其中，第 ![[公式]](https://www.zhihu.com/equation?tex=m) 棵树对应的响应值为（损失函数的负梯度，即伪残差)：

![image-20210719091614388](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091614388.png)

对于生成的决策树，计算各个叶子节点的最佳残差拟合值为：

![image-20210719091629182](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091629182.png)

由于上式没有闭式解（closed form solution），我们一般使用近似值代替：

![image-20210719091641826](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719091641826.png)

### GBDT二分类算法训练过程

1. 初始化第一个弱分类器$F_0$(x)：

   ![image-20210719093404050](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210719093404050.png)

   其中， ![[公式]](https://www.zhihu.com/equation?tex=P%28Y%3D1%7Cx%29) 是训练样本中 ![[公式]](https://www.zhihu.com/equation?tex=y%3D1) 的比例，利用先验信息来初始化学习器。

2. **对于建立** ![[公式]](https://www.zhihu.com/equation?tex=M) **棵分类回归树** ![[公式]](https://www.zhihu.com/equation?tex=m%3D1%2C2%2C...%2CM) :

* 对 ![[公式]](https://www.zhihu.com/equation?tex=i+%3D+1%2C2%2C...%2CN) ，计算第 ![[公式]](https://www.zhihu.com/equation?tex=m) 棵树对应的响应值（损失函数的负梯度，即伪残差）：

  ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bm%2Ci%7D+%3D+-+%5Cleft%5B+%5Cfrac%7B%5Cpartial+L%28y_%7Bi%7D%2CF%28x_%7Bi%7D%29%29%7D%7B%5Cpartial+F%28x%29%7D+%5Cright%5D_%7BF%28x%29%3DF_%7Bm-1%7D%28x%29+%7D+%3D++y_i+-+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-F%28x_i%29%7D%7D)

* 对于![[公式]](https://www.zhihu.com/equation?tex=i+%3D+1%2C2+%2C+%5Cdots+%2C+N) ，利用CART回归树拟合数据 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28x_%7Bi%7D%2C+r_%7Bm%2C+i%7D%5Cright%29) ，得到第 ![[公式]](https://www.zhihu.com/equation?tex=m) 棵回归树，其对应的叶子节点区域为 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bm%2C+j%7D) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=j%3D1%2C2%2C+%5Cdots%2C+J_%7Bm%7D) ，且 ![[公式]](https://www.zhihu.com/equation?tex=J_%7Bm%7D) 为第![[公式]](https://www.zhihu.com/equation?tex=m)棵回归树叶子节点的个数。

* 对于![[公式]](https://www.zhihu.com/equation?tex=J_%7Bm%7D) 个叶子节点区域 ![[公式]](https://www.zhihu.com/equation?tex=j%3D1%2C2%2C+%5Cdots%2C+J_%7Bm%7D)，计算出最佳拟合值：

  ![[公式]](https://www.zhihu.com/equation?tex=c_%7Bm%2Cj%7D+%3D+%5Cfrac%7B%5Csum_%7Bx_%7Bi%7D+%5Cin+R_%7Bm%2Cj%7D%7D%5E%7B%7D%7Br_%7Bm%2Ci%7D%7D%7D%7B%5Csum_%7Bx_%7Bi%7D+%5Cin+R_%7Bm%2Cj%7D%7D%5E%7B%7D%7B%28y_%7Bi%7D-r_%7Bm%2Ci%7D%29%281-y_%7Bi%7D%2B+r_%7Bm%2Ci%7D%29+%7D%7D)

* 更新强学习器 ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bm%7D%28x%29) ：

  ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bm%7D%28x%29%3DF_%7Bm-1%7D%28x%29%2B%5Csum_%7Bj%3D1%7D%5E%7BJ_%7Bm%7D%7D+c_%7Bm%2C+j%7D+I%5Cleft%28x+%5Cin+R_%7Bm%2Cj%7D%5Cright%29)

3. **得到最终的强学习器 ![[公式]](https://www.zhihu.com/equation?tex=F_%7BM%7D%28x%29) 的表达式：**

   ![[公式]](https://www.zhihu.com/equation?tex=F_%7BM%7D%28x%29%3DF_%7B0%7D%28x%29%2B%5Csum_%7Bm%3D1%7D%5E%7BM%7D+%5Csum_%7Bj%3D1%7D%5E%7BJ_%7Bm%7D%7D+c_%7Bm%2Cj%7D+I%5Cleft%28x+%5Cin+R_%7Bm%2C+j%7D%5Cright%29)

从以上过程中可知，除了由损失函数引起的负梯度计算和叶子节点的最佳残差拟合值的计算不同，二元GBDT分类和GBDT回归算法过程基本相似。那二元GBDT是如何做分类呢？

将逻辑回归的公式进行整理，我们可以得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Clog+%5Cfrac%7Bp%7D%7B1-p%7D+%3D+%5Ctheta%5E%7BT%7Dx) ，其中![[公式]](https://www.zhihu.com/equation?tex=p%3DP%28Y%3D1%7Cx%29)，也就是将给定输入 ![[公式]](https://www.zhihu.com/equation?tex=x) 预测为正样本的概率。逻辑回归用一个线性模型去拟合 ![[公式]](https://www.zhihu.com/equation?tex=Y%3D1%7Cx) 这个事件的对数几率（odds）![[公式]](https://www.zhihu.com/equation?tex=%5Clog+%5Cfrac%7Bp%7D%7B1-p%7D)。二元GBDT分类算法和逻辑回归思想一样，用一系列的梯度提升树去拟合这个对数几率，其分类模型可以表达为：

![[公式]](https://www.zhihu.com/equation?tex=P%28Y%3D1%7Cx%29+%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-F_%7BM%7D%28x%29%7D%7D)

## 二分类算法实例

### **（1）数据集介绍**

训练集如下表所示，一组数据的特征有年龄和体重，把身高大于1.5米作为分类边界，身高大于1.5米的令标签为1，身高小于等于1.5米的令标签为0，共有4组数据。

![img](https://pic2.zhimg.com/80/v2-f985b918beb9465e1f1d63d4df26653d_1440w.jpg)

测试数据如下表所示，只有一组数据，年龄为25、体重为65，我们用在训练集上训练好的GBDT模型预测该组数据的身高是否大于1.5米？

![img](https://pic2.zhimg.com/80/v2-b86998f87dd21433d45247347081b561_1440w.png)

### **（2）模型训练阶段**

**参数设置：**

- 学习率：learning_rate = 0.1
- 迭代次数：n_trees = 5
- 树的深度：max_depth = 3

**1）初始化弱学习器：**

![[公式]](https://www.zhihu.com/equation?tex=F_%7B0%7D%28x%29%3Dlog%5Cfrac%7BP%28Y%3D1%7Cx%29%7D%7B1-P%28Y%3D1%7Cx%29%7D%3Dlog+%5Cfrac%7B2%7D%7B2%7D+%3D+0)

**2）对于建立M棵分类回归树**![[公式]](https://www.zhihu.com/equation?tex=m+%3D+1%2C2+%2C+%5Cdots+%2C+M)**：**

由于我们设置了迭代次数：n_trees=5，这就是设置了M=5。

**首先计算负梯度**，根据上文损失函数为对数损失时，负梯度（即伪残差、近似残差）为：

![[公式]](https://www.zhihu.com/equation?tex=r_%7Bm%2Ci%7D+%3D+-+%5Cleft%5B+%5Cfrac%7B%5Cpartial+L%28y_%7Bi%7D%2CF%28x_%7Bi%7D%29%29%7D%7B%5Cpartial+F%28x%29%7D+%5Cright%5D_%7BF%28x%29%3DF_%7Bm-1%7D%28x%29+%7D+%3D++y_i+-+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-F%28x_i%29%7D%7D)

我们知道梯度提升类算法，其关键是利用损失函数的负梯度的值作为回归问题提升树算法中的残差的近似值，拟合一个回归树。这里，为了称呼方便，我们把负梯度叫做残差。

现将残差的计算结果列表如下：

![img](https://pic3.zhimg.com/80/v2-5f8b01323172e9186389745dba59cfba_1440w.jpg)

此时将残差作为样本的标签来训练弱学习器 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B1%7D%28x%29) ，即下表数据：

![img](https://pic3.zhimg.com/80/v2-3665bd298d997938eeae08de504962a2_1440w.jpg)

**接着寻找回归树的最佳划分节点**，遍历每个特征的每个可能取值。从年龄特征值为5开始，到体重特征为70结束，分别计算分裂后两组数据的平方损失（Square Error），![[公式]](https://www.zhihu.com/equation?tex=SE_%7Bl%7D) 为左节点的平方损失， ![[公式]](https://www.zhihu.com/equation?tex=SE_%7Br%7D) 为右节点的平方损失，找到使平方损失和 ![[公式]](https://www.zhihu.com/equation?tex=SE_%7Bsum%7D+%3D+SE_%7Bl%7D%2BSE_%7Br%7D) 最小的那个划分节点，即为最佳划分节点。

例如：以年龄7为划分节点，将小于7的样本划分为到左节点，大于等于7的样本划分为右节点。左节点包括 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B0%7D)，右节点包括样本 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B1%7D%2Cx_%7B2%7D%2Cx_%7B3%7D)， ![[公式]](https://www.zhihu.com/equation?tex=SE_%7Bl%7D%3D0)， ![[公式]](https://www.zhihu.com/equation?tex=SE_%7Br%7D%3D0.667)，![[公式]](https://www.zhihu.com/equation?tex=SE_%7Bsum%7D%3D0.667) ，所有可能的划分情况如下表所示：

![img](https://pic4.zhimg.com/80/v2-1a902785ce01139828a3449e8b8f7993_1440w.jpg)

以上划分点的总平方损失最小为**0.000，**有两个划分点：年龄21和体重60，所以随机选一个作为划分点，这里我们选**年龄21**。现在我们的第一棵树长这个样子：

![img](https://pic3.zhimg.com/80/v2-46ec3591fbaafaa1ce80ae66501e9296_1440w.jpg)

我们设置的参数中树的深度max_depth=3，现在树的深度只有2，需要再进行一次划分，这次划分要对左右两个节点分别进行划分，但是我们在生成树的时候，设置了三个树继续生长的条件：

- **深度没有到达最大。树的深度设置为3，意思是需要生长成3层。**
- **点样本数 >= min_samples_split**
- ***此节点上的样本的标签值不一样（如果值一样说明已经划分得很好了，不需要再分）（本程序满足这个条件，因此树只有2层）\***

最终我们的第一棵回归树长下面这个样子：

![img](https://pic4.zhimg.com/80/v2-723c248d3a437ece72fafb8ce14ea9bf_1440w.jpg)

此时我们的树满足了设置，还需要做一件事情，给这棵树的每个叶子节点分别赋一个参数![[公式]](https://www.zhihu.com/equation?tex=c)，来拟合残差。

![[公式]](https://www.zhihu.com/equation?tex=c_%7B1%2Cj%7D+%3D+%5Cfrac%7B%5Csum_%7Bx_%7Bi%7D+%5Cin+R_%7B1%2Cj%7D%7D%5E%7B%7D%7Br_%7B1%2Ci%7D%7D%7D%7B%5Csum_%7Bx_%7Bi%7D+%5Cin+R_%7B1%2Cj%7D%7D%5E%7B%7D%7B%28y_%7Bi%7D-r_%7B1%2Ci%7D%29%281-y_%7Bi%7D%2B+r_%7B1%2Ci%7D%29+%7D%7D)

根据上述划分结果，为了方便表示，规定从左到右为第1,2个叶子结点，其计算值过程如下：

![[公式]](https://www.zhihu.com/equation?tex=%28x_%7B0%7D%2Cx_%7B1%7D%5Cin+R_%7B1%2C1%7D%29%2C+%5Cqquad+c_%7B1%2C1%7D%3D-2.0)

![[公式]](https://www.zhihu.com/equation?tex=%28x_%7B2%7D%2C+x_%7B3%7D%5Cin+R_%7B1%2C2%7D%29%2C+%5Cqquad+c_%7B1%2C2%7D%3D2.0)

此时的第一棵树长下面这个样子：

![img](https://pic3.zhimg.com/80/v2-6306341d6ef4848a017c03d8496ed6b6_1440w.jpg)

接着更新强学习器，需要用到学习率：learning_rate=0.1，用![[公式]](https://www.zhihu.com/equation?tex=lr)表示。更新公式为：

![[公式]](https://www.zhihu.com/equation?tex=F_%7B1%7D%28x%29%3DF_%7B0%7D%28x%29%2Blr+%2A+%5Csum_%7Bj%3D1%7D%5E%7B2%7D+c_%7B1%2C+j%7D+I%5Cleft%28x+%5Cin+R_%7B1%2Cj%7D%5Cright%29)

为什么要用学习率呢？这是**Shrinkage**的思想，如果每次都全部加上拟合值 ![[公式]](https://www.zhihu.com/equation?tex=c) ，即学习率为1，很容易一步学到位导致GBDT过拟合。

**重复此步骤，直到** **![[公式]](https://www.zhihu.com/equation?tex=m%3E5)** **结束，最后生成5棵树。**

下面将展示每棵树最终的结构，这些图都是我GitHub上的代码生成的，感兴趣的同学可以去运行一下代码。[https://github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning/GBDT_GradientBoostingBinaryClassifier](https://link.zhihu.com/?target=https%3A//github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning/GBDT_GradientBoostingBinaryClassifier)

**第一棵树：**

![img](https://pic3.zhimg.com/80/v2-6306341d6ef4848a017c03d8496ed6b6_1440w.jpg)

**第二棵树：**

![img](https://pic1.zhimg.com/80/v2-73b99303ccecdc78f59d9da5e7523de4_1440w.jpg)

**第三棵树：**

![img](https://pic2.zhimg.com/80/v2-c24b8c302fdb469ab169bc2425d336a5_1440w.jpg)

**第四棵树：**

![img](https://pic2.zhimg.com/80/v2-7e26f3b72816524240155e8b672e63b9_1440w.jpg)

**第五棵树：**

![img](https://pic2.zhimg.com/80/v2-9640af653aa523ef5cc2d5ffcf8a9249_1440w.jpg)

**3）得到最后的强学习器：**

![[公式]](https://www.zhihu.com/equation?tex=F_%7B5%7D%28x%29%3DF_%7B0%7D%28x%29%2B%5Csum_%7Bm%3D1%7D%5E%7B5%7D+%5Csum_%7Bj%3D1%7D%5E%7B2%7D+c_%7Bm%2Cj%7D+I%5Cleft%28x+%5Cin+R_%7Bm%2C+j%7D%5Cright%29)

### **（3）模型预测阶段**

- ![[公式]](https://www.zhihu.com/equation?tex=F_%7B0%7D%28x%29+%3D+0)
- 在 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B1%7D%28x%29) 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**2.0000**。
- 在 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B2%7D%28x%29) 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.8187**。
- 在 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B3%7D%28x%29) 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.6826**。
- 在 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B4%7D%28x%29) 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.5769**。
- 在 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B5%7D%28x%29) 中，测试样本的年龄为25，大于划分节点21岁，所以被预测为**1.4927**。

**最终预测结果为：**

![[公式]](https://www.zhihu.com/equation?tex=F%28x%29+%3D+0.0000+%2B+0.1+%2A+%282.0000%2B1.8187%2B1.6826%2B1.5769%2B1.4927%29%3D0.8571)

![[公式]](https://www.zhihu.com/equation?tex=P%28Y%3D1%7Cx%29+%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-F%28x%29%7D%7D%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-0.8571%7D%7D+%3D+0.7021)

