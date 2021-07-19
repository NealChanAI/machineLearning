# GBDT多分类

## Softmax回归的对数损失函数

当使用逻辑回归处理多标签的分类问题时，如果一个样本只对应于一个标签，我们可以假设每个样本属于不同标签的概率服从于几何分布，使用多项逻辑回归（Softmax Regression）来进行分类：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Baligned%7D+P%28Y%3Dy_%7Bi%7D%7Cx%29+%26%3D+h_%7B%5Ctheta%7D%28x%29+%5Cbegin%7Bbmatrix%7D+P%28Y%3D1%7Cx%3B%5Ctheta+%29%5C%5C+P%28Y%3D2%7Cx%3B%5Ctheta%29+%5C%5C+.+%5C%5C+.+%5C%5C+.+%5C%5C+P%28Y%3Dk%7Cx%3B%5Ctheta%29+%5Cend%7Bbmatrix%7D+%5C%5C+%26%3D%5Cfrac%7B1%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bk%7D%7Be%5E%7B%5Ctheta_%7Bj%7D%5E%7BT%7Dx%7D%7D%7D+%5Cbegin%7Bbmatrix%7D+e%5E%7B%5Ctheta%5E%7BT%7D_%7B1%7Dx%7D+%5C%5C+e%5E%7B%5Ctheta%5E%7BT%7D_%7B2%7Dx%7D+%5C%5C+.+%5C%5C+.+%5C%5C+.+%5C%5C+e%5E%7B%5Ctheta%5E%7BT%7D_%7Bk%7Dx+%7D+%5Cend%7Bbmatrix%7D+%5Cend%7Baligned%7D+%5Cend%7Bequation%7D)

当存在样本可能属于多个标签的情况时，我们可以训练 ![[公式]](https://www.zhihu.com/equation?tex=k) 个二分类的逻辑回归分类器。第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个分类器用以区分每个样本是否可以归为第 ![[公式]](https://www.zhihu.com/equation?tex=i) 类，训练该分类器时，需要把标签重新整理为"第 ![[公式]](https://www.zhihu.com/equation?tex=i) 类标签"与"非第 ![[公式]](https://www.zhihu.com/equation?tex=i) 类标签”两类。通过这样的办法，我们就解决了每个样本可能拥有多个标签的情况。

在二分类的逻辑回归中，对输入样本 ![[公式]](https://www.zhihu.com/equation?tex=x) 分类结果为类别1和0的概率可以写成下列形式：

![[公式]](https://www.zhihu.com/equation?tex=P%28Y%3Dy%7Cx%3B%5Ctheta%29+%3D+%28h_%7B%5Ctheta%7D%28x%29%29%5E%7By%7D%281-h_%7B%5Ctheta%7D%28x%29%29%5E%7B1-y%7D)

其中， ![[公式]](https://www.zhihu.com/equation?tex=h_%7B%5Ctheta%7D%28x%29+%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5E%7BT%7Dx%7D%7D) 是模型预测的概率值， ![[公式]](https://www.zhihu.com/equation?tex=y) 是样本对应的类标签。

将问题泛化为更一般的多分类情况：

![[公式]](https://www.zhihu.com/equation?tex=P%28Y%3Dy_%7Bi%7D%7Cx%3B%5Ctheta%29+%3D%5Cprod_%7Bi%3D1%7D%5E%7BK%7DP%28y_%7Bi%7D%7Cx%29%5E%7By_%7Bi%7D%7D+%3D+%5Cprod_%7Bi%3D1%7D%5E%7BK%7Dh_%7B%5Ctheta%7D%28x%29%5E%7By_%7Bi%7D%7D)

由于连乘可能导致最终结果接近0的问题，一般对似然函数取对数的负数，变成最小化对数似然函数。

![[公式]](https://www.zhihu.com/equation?tex=-logP%28Y%3Dy_%7Bi%7D%7Cx%3B%5Ctheta%29+%3D-log+%5Cprod_%7Bi%3D1%7D%5E%7BK%7DP%28y_%7Bi%7D%7Cx%29%5E%7By_%7Bi%7D%7D+%3D+-%5Csum_%7Bi%3D1%7D%5E%7BK%7D%7By_%7Bi%7Dlog%28h_%7B%5Ctheta%7D%28x%29%29%7D)

## GBDT多分类原理

将GBDT应用于二分类问题需要考虑逻辑回归模型，同理，对于GBDT多分类问题则需要考虑以下Softmax模型：

![[公式]](https://www.zhihu.com/equation?tex=P%28y%3D1%7Cx%29+%3D+%5Cfrac%7Be%5E%7BF_1%28x%29%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ek+e%5E%7BF_i%28x%29%7D%7D%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=P%28y%3D2%7Cx%29+%3D+%5Cfrac%7Be%5E%7BF_2%28x%29%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ek+e%5E%7BF_i%28x%29%7D%7D%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=...+...%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=P%28y%3Dk%7Cx%29+%3D+%5Cfrac%7Be%5E%7BF_k%28x%29%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Ek+e%5E%7BF_i%28x%29%7D%7D%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=F_1+...) ![[公式]](https://www.zhihu.com/equation?tex=F_k) 是 ![[公式]](https://www.zhihu.com/equation?tex=k) 个不同的CART回归树集成。每一轮的训练实际上是训练了 ![[公式]](https://www.zhihu.com/equation?tex=k) 棵树去拟合softmax的每一个分支模型的负梯度。softmax模型的单样本损失函数为：

![[公式]](https://www.zhihu.com/equation?tex=loss+%3D+-%5Csum_%7Bi%3D1%7D%5Ek+y_i+%5Clog+P%28y_i%7Cx%29+%3D+-%5Csum_%7Bi%3D1%7D%5Ek+y_i+%5Clog+%5Cfrac%7Be%5E%7BF_i%28x%29%7D%7D%7B%5Csum_%7Bj%3D1%7D%5Ek+e%5E%7BF_j%28x%29%7D%7D%5C%5C)

这里的 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5C+%28i%3D1...k%29) 是样本label在k个类别上作one-hot编码之后的取值，只有一维为1，其余都是0。由以上表达式不难推导：

![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7B%5Cpartial+loss%7D%7B%5Cpartial+F_i%7D+%3D+y_i+-+%5Cfrac%7Be%5E%7BF_i%28x%29%7D%7D%7B%5Csum_%7Bj%3D1%7D%5Ek+e%5E%7BF_j%28x%29%7D%7D+%3D+y_i+-+p%28y_%7Bi%7D%7Cx%29%5C%5C)

可见，这 ![[公式]](https://www.zhihu.com/equation?tex=k) 棵树同样是拟合了样本的真实标签与预测概率之差，与GBDT二分类的过程非常类似。下图是Friedman在论文中对GBDT多分类给出的伪代码：

![img](https://img-blog.csdn.net/20180129182053132?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjIyMzg1MzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

根据上面的伪代码具体到多分类这个任务上面来，我们假设总体样本共有 ![[公式]](https://www.zhihu.com/equation?tex=K) 类。来了一个样本 ![[公式]](https://www.zhihu.com/equation?tex=x) ，我们需要使用GBDT来判断 ![[公式]](https://www.zhihu.com/equation?tex=x) 属于样本的哪一类。

**第一步我们在训练的时候，是针对样本 ![[公式]](https://www.zhihu.com/equation?tex=x) 每个可能的类都训练一个分类回归树。**举例说明，目前样本有三类，也就是 ![[公式]](https://www.zhihu.com/equation?tex=K%3D3) ，样本 ![[公式]](https://www.zhihu.com/equation?tex=x) 属于第二类。那么针对该样本的分类标签，其实可以用一个三维向量 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%2C0%5D) 来表示。 ![[公式]](https://www.zhihu.com/equation?tex=0) 表示样本不属于该类， ![[公式]](https://www.zhihu.com/equation?tex=1) 表示样本属于该类。由于样本已经属于第二类了，所以第二类对应的向量维度为 ![[公式]](https://www.zhihu.com/equation?tex=1) ，其它位置为 ![[公式]](https://www.zhihu.com/equation?tex=0) 。

**针对样本有三类的情况，我们实质上在每轮训练的时候是同时训练三颗树。**第一颗树针对样本 ![[公式]](https://www.zhihu.com/equation?tex=x) 的第一类，输入为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C0%29) 。第二颗树输入针对样本 ![[公式]](https://www.zhihu.com/equation?tex=x) 的第二类，输入为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C1%29) 。第三颗树针对样本 ![[公式]](https://www.zhihu.com/equation?tex=x+) 的第三类，输入为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C0%29) 。这里每颗树的训练过程其实就CART树的生成过程。在此我们参照CART生成树的步骤即可解出三颗树，以及三颗树对 ![[公式]](https://www.zhihu.com/equation?tex=x) 类别的预测值 ![[公式]](https://www.zhihu.com/equation?tex=+F_%7B1%7D%28x%29%2C+F_%7B2%7D%28x%29%2C+F_%7B3%7D%28x%29) , 那么在此类训练中，我们仿照多分类的逻辑回归 ，使用Softmax 来产生概率，则属于类别 ![[公式]](https://www.zhihu.com/equation?tex=1) 的概率为：

![[公式]](https://www.zhihu.com/equation?tex=p_%7B1%7D%28x%29+%3D+%5Cfrac%7Bexp%28F_%7B1%7D%28x%29%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7B3%7D%7Bexp%28F_%7Bk%7D%28x%29%29%7D%7D+%5C%5C) 并且我们可以针对类别 ![[公式]](https://www.zhihu.com/equation?tex=1) 求出残差 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D_%7B1%7D%3D0%E2%88%92p_%7B1%7D%28x%29) ；类别 ![[公式]](https://www.zhihu.com/equation?tex=2+) 求出残差 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D_%7B2%7D%3D0%E2%88%92p_%7B2%7D%28x%29) ；类别 ![[公式]](https://www.zhihu.com/equation?tex=3+) 求出残差 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D_%7B3%7D%3D0%E2%88%92p_%7B3%7D%28x%29) 。

然后开始第二轮训练，针对第一类输入为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C%5Ctilde%7By%7D_%7B1%7D%29) , 针对第二类输入为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C%5Ctilde%7By%7D_%7B2%7D%29) ，针对第三类输入为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C%5Ctilde%7By%7D_%7B3%7D%29) 。继续训练出三颗树。一直迭代M轮。每轮构建3颗树。

当 ![[公式]](https://www.zhihu.com/equation?tex=K%3D3) 时，我们其实应该有三个式子：

![[公式]](https://www.zhihu.com/equation?tex=F_%7B1M%7D%7B%28x%29%7D%3D%5Csum_%7Bm%3D1%7D%5E%7BM%7D%7Bc_%7B1m%7DI%28x%5Cepsilon+R_%7B1m%7D%29%7D++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=F_%7B2M%7D%7B%28x%29%7D%3D%5Csum_%7Bm%3D1%7D%5E%7BM%7D%7Bc_%7B2m%7DI%28x%5Cepsilon+R_%7B2m%7D%29%7D++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=F_%7B3M%7D%7B%28x%29%7D%3D%5Csum_%7Bm%3D1%7D%5E%7BM%7D%7Bc_%7B3m%7DI%28x%5Cepsilon+R_%7B3m%7D%29%7D++%5C%5C)

当训练完以后，新来一个样本 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B1%7D) ，我们要预测该样本类别的时候，便可以有这三个式子产生三个值 ![[公式]](https://www.zhihu.com/equation?tex=F_%7B1M%7D%2CF_%7B2M%7D%2CF_%7B3M%7D) 。样本属于某个类别的概率为：

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bi%7D%28x%29%3D%5Cfrac%7Bexp%28F_%7BiM%7D%7B%28x%29%7D%29%7D%7B%5Csum_%7Bk%3D+1%7D%5E%7B3%7Dexp%28F_%7BkM%7D%7B%28x%29%7D%29+%7D+%5C%5C)

个人理解:

训练过程:

1. 初始化$F_0$=0
2. 将一个样本$x_i$的标签y进行one-hot encoding(K维), 然后将one-hot向量分别作为标签值输入到K个分类器中
3. 根据$F_i$计算softmax概率值p, 同一轮的每棵树求得一个概率值
4. 根据损失函数对$F_i$的偏导公式计算负梯度
5. 计算每个叶子节点的输出值, 并加到强分类器中, 