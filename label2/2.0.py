import numpy as np
##将文本数据转化为矩阵
F1 = open("iris.txt", "r") #转化为numpy数组
List = F1.readlines()
list0 = []
for i in range(len(List)):
  list1 = List[i].strip().split(",")  # 每一行以，为分隔的数据后是一个列表
  list1.pop()#去掉最后一列属性
  list0.append(list1)  # 加入list0
a=np.array(list0)#转化为np数组
a=a.astype(float)#转换为浮点类型
#计算核矩阵
K=np.zeros((len(a),len(a)))#生成一个长宽为len（a）的空矩阵
for m in range(len(a)):
  for n in range(len(a)):
     K[m][n]=np.dot(a[m],a[n])
#中心化矩阵
mu=np.mean(K)#计算k矩阵均值
Center=K-mu

#标准化矩阵
sigma=np.std(K)
Normal=(K-mu)/sigma

##齐次二次核矩阵计算
Marix=np.zeros((len(a),10))
for i in range (len(a)):
 for j in range(4):
   Marix[i][j] = a[i][j]*a[i][j] #前四个属性为平方
 Marix[i][4] = np.sqrt(2) * a[i][0] * a[i][1]
 Marix[i][5] = np.sqrt(2) * a[i][0] * a[i][2]
 Marix[i][6] = np.sqrt(2) * a[i][0] * a[i][3]
 Marix[i][7] = np.sqrt(2) * a[i][1] * a[i][2]
 Marix[i][8] = np.sqrt(2) * a[i][1] * a[i][3]
 Marix[i][9] = np.sqrt(2) * a[i][2] * a[i][3]
##齐次二次矩阵标准化
muM=np.mean(Marix)#计算均值mu
sigmaM=np.std(Marix)#计算标准差
NormalM=(Marix-muM)/sigmaM#标准化
##中心化
M=Marix-muM
print(M)