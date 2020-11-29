import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

# 读取文件数据
fp = open('train_data.txt', 'r')

# 迭代次数 整体误差 正确率
train_iterations = []
train_loss = []
test_accuracy = []

# 解析数据
for line in fp.readlines():
    con = line.strip('\n').split(',')
    print(con)
    train_iterations.append(int(con[0]))
    train_loss.append(float(con[1]))
    test_accuracy.append(float(con[2]))

# 绘制曲线图
host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
par1 = host.twinx()

# 设置类标
host.set_xlabel("iterations")
host.set_ylabel("loss")
par1.set_ylabel("validation accuracy")

# 绘制曲线
p1, = host.plot(train_iterations, train_loss, "b-", label="training loss")
p2, = host.plot(train_iterations, train_loss, ".") #曲线点
p3, = par1.plot(train_iterations, test_accuracy, label="validation accuracy")
p4, = par1.plot(train_iterations, test_accuracy, "1")

# 设置图标
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=5)

# 设置颜色
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p3.get_color())

# 设置范围
host.set_xlim([-10, 1000])

plt.draw()
plt.show()
