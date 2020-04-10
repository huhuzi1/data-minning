import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

f = open("2.txt", "r")
row = f.readlines()
list = []
for i in range(len(row)):
    column_list = row[i].strip().split(",")  # 每一行以，为分隔符split后是一个列表
    column_list.pop()#去掉最后一行属性g
    list.append(column_list)  # 加入list_source
a=np.array(list)#转化为np数组
a=a.astype(float)#转换为浮点类型
MeanVector=np.mean(a,axis=0)#均值向量
center=a-MeanVector#中心化
innerProduct=np.dot(center.T,center)
print(innerProduct/len(center))#求内积
Kroneckerproduct=0
for i in range(len(center)):
    Kroneckerproduct = Kroneckerproduct+center[i].reshape(len(center[0]),1)*center[i]
print(Kroneckerproduct/len(center))#求外积
t=center.T#通过中心化后的向量计算属性1和2的夹角
corr=np.corrcoef(t[0],t[1])#计算第一列属性和第二列属性相关性
print(corr[0][1])
picture = plt.figure()
ax1 = picture.add_subplot(111)  #设置标题
ax1.set_title("Correlation scatter plot")
plt.scatter(t[0],t[1])
plt.xlabel('Attributes 1')  #设置X轴标签
plt.ylabel('Attributes 2') #设置Y轴标签
plt.show()
# 正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
#计算mu和sigma
mu=np.mean(a,axis=0)[0]#计算第一列均值即mu
sigma=np.var(a.T[0])#计算第一列方差及sigma
fig = plt.figure()
ax1 = picture.add_subplot(111)
ax1.set_title("Probability density function")
# Python实现正态分布
# 绘制正态分布概率密度函数
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
y_sig = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.vlines(mu, 0, np.exp(-(mu - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma), colors="c",
           linestyles="dashed")
plt.vlines(mu + sigma, 0, np.exp(-(mu + sigma - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma),
           colors="k", linestyles="dotted")
plt.vlines(mu - sigma, 0, np.exp(-(mu - sigma - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma),
           colors="k", linestyles="dotted")
plt.xticks([mu - sigma, mu, mu + sigma], ['μ-σ', 'μ', 'μ+σ'])
plt.xlabel('Attributes 1')
plt.ylabel('Attributes 2')
plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f' % (mu, sigma))
plt.grid(True)
plt.show()
#求每一列的方差
list=[]
for i in range(len(b[0])):
    list.append(np.var(b.T[i]))
print(list)
maxIndex=list.index(max(list))
minIndex=list.index(min(list))
print(maxIndex+1)
print(minIndex+1)
#求矩阵两列协方差
Cov={}
for i in range(9):
    for j in range(i+1,10):
        st=str(i+1)+'-'+str(j+1)
        Cov[st]= np.cov(a.T[i],a.T[j])[0][1]#遍历求协方差
print(Cov)
print(min(Cov, key=Cov.get))#取最小值打印
print(max(Cov, key=Cov.get))#取最大值打印