import cv2
import subprocess
import torch
import PyHook3
import pythoncom
import threading

data = []
data_not_using = True

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.h2h = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.h2h(x))
        x = torch.relu(self.h2h(x))
        x = torch.relu(self.h2h(x))
        x = torch.relu(self.h2h(x))
        x = torch.relu(self.h2h(x))
        x = torch.relu(self.h2h(x))
        x = self.out(x)
        return x

net = Net(17, 20, 2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()

def detect():
    global data, data_not_using
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()

        # 输出摄像头文件至test.jpg
        cv2.imwrite("test.jpg", frame)

        # 调用OpenFace处理test.jpg输出test文件夹下的test.csv
        p = subprocess.Popen(".\\OpenFace\\FeatureExtraction.exe -f ./test.jpg -out_dir ./test", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sout = p.stdout.readlines()
        sout.clear()

        # 读取test.csv数据至data
        with open('.\\test\\test.csv', 'r') as f:
            tempStr = f.read().split('\n')[1].split(',')
        innerDNU = data_not_using
        if float(tempStr[3]) >= 0.90 and innerDNU:
            data_not_using = False
            data.clear()
            for i in tempStr:
                data.append(float(i))
            data.pop(0)
            data.pop(1)
            data.pop(2)
            data.pop(3)
            data.pop(4)
            initData()
            data_not_using = True

        # 检测Q键按下退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()

def toOne(a):
    return (a - a.min()) / (a.max() - a.min())

# 数据预处理函数
def initData():
    global data, data_not_using
    Position = toOne(torch.tensor([data[457], data[252], data[593]]))
    gazePoint = toOne(torch.tensor(data[0:6]))
    gazeAngle = toOne(torch.tensor(data[6:8]))
    # eye3d = toOne(torch.tensor(data[120:288]))
    poseT = toOne(torch.tensor(data[288:291]))
    poseR = toOne(torch.tensor(data[291:294]))
    data.clear()
    tempData = torch.cat((Position, gazePoint, gazeAngle, poseT, poseR), 0)
    tempList = tempData.tolist()
    for i in tempList:
        data.append(i)


# 学习器训练函数
def trainModel(faceData, mousePoint):
    global data, data_not_using
    out = net(faceData)
    loss = loss_func(out, mousePoint)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return out


# 鼠标点击时进行BP网络训练,并输出预测结果和真实结果
def onMouseEvent(event):
    global data, data_not_using
    a = event.Position
    outLayer = torch.tensor([float(a[0]/1920), float(a[1])/1080], dtype=torch.float)
    inLayer = torch.tensor(data, dtype=torch.float)
    out = trainModel(inLayer, outLayer)
    print(out, '\n', outLayer)
    return 0

def mouseThread():
    global data, data_not_using
    hm = PyHook3.HookManager()
    hm.MouseAllButtonsDown = onMouseEvent
    hm.HookMouse()
    pythoncom.PumpMessages()

if __name__ == "__main__":
    data_not_using = True
    threading.Thread(target=detect).start()
    threading.Thread(target=mouseThread).start()

