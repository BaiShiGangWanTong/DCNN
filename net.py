import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

#制作假的数据集
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y = torch.asin(x) + 0.2*torch.rand(x.size())


class Net(torch.nn.Module):
    def __init__(self, n_features, hiddens, o_features):
        super(Net, self).__init__()  # 这一行必须要加
        self.hidden = torch.nn.Linear(n_features, hiddens)
        self.predict = torch.nn.Linear(hiddens, o_features)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.predict(x)

        return x

#构建一个网络
net = Net(n_features=1, hiddens=128, o_features=1)
print(net)
'''
Net(
  (hidden): Linear(in_features=1, out_features=128, bias=True)
  (predict): Linear(in_features=128, out_features=1, bias=True)
)
'''

#设置优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
#误差计算
loss_func = torch.nn.MSELoss()

#训练
for t in range(100):
    #预测值
    prediction = net(x)
    loss = loss_func(prediction, y)
    #清空梯度参数
    optimizer.zero_grad()
    #误差反向传播更新参数
    loss.backward()
    #更新网络参数
    optimizer.step()

#可视化最后拟合出来的曲线
plt.figure()
plt.scatter(x.data.numpy(), y.data.numpy(), color='blue')
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color':  'red'})
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
