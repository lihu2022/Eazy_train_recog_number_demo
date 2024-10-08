import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import sys
import time
from PIL import Image, ImageDraw
import numpy as np
import tkinter as tk
import io
# import os
# import numpy as np

# xp = np.arange(0, 5, 0.1)
# yp = np.sin(xp)
# plt.plot(xp, yp)
# plt.show()

class FC_Net(torch.nn.Module):
    #初始化全连接神经网络，总参数。
    #进行修改参数，谨记神经网络的偏置项是每一层的输出层的个数，而不是跟随比例系数一样是对应输入神经元的个数，进行修改，只是为了保证模型可以不一定要过零点，从而引入更细致的拟合。
    #total params：50176+8192+640+64+64+64+10=59210
    def __init__(self):
        super().__init__()
        #输入层：将像素为28*28分辨率的照片展成一纵列，全连接到第二层network为64个神经元，偏置项是第二层
        #params：50176+64
        self.fc1 = torch.nn.Linear(28*28, 64)
        #中间层：设置两层中间层都为64神经元，不改变维度大小
        #params：8192+64+64
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        #输出层：将神经元连接到输出层，输出十个数字的概率。
        #params:640+10
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        #非线性化relu，考虑是否可以只非线性化一层进行简化？？##
        #re：不太建议，因为要使得非线性的程度高，一般在每个隐藏层后都添加非线性函数
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        #softmax进行概率分化
        x = torch.nn.functional.softmax(self.fc4(x), dim = 1)

        return x
    


class CNN_Net(torch.nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        # 输入图像是单通道像素值为28x28 图像
        # 定义两个卷积层
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        # self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 卷积后，图像的尺寸并未改变
        
        # 仅为示例，没有考虑卷积之后的具体大小变化，
        # 可能需要调整参数以符合实际大小。
        # 加入池化层后图像尺寸为：[batch, 64, 14, 14] 
        # 接下来使用线性层，需要先将数据展平
        # self.fc1 = torch.nn.Linear(64 * 14 * 14, 64)
        self.fc1 = torch.nn.Linear(4 * 14 * 14, 64)
        # self.fc2 = torch.nn.Linear(64, 64)
        # self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 应用卷积层和 ReLU 非线性
        # x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        # x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
        
        # 展平
        # x = x.view(-1, 64 * 14 * 14)
        x = x.view(-1, 4 * 14 * 14)
        
        # 应用全连接层和 ReLU 非线性
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        # x = torch.nn.functional.relu(self.fc3(x))
        
        # 最后一层不进行 ReLU，直接 softmax
        x = torch.nn.functional.softmax(self.fc4(x), dim=1)

        return x
    

def Get_Download_data(is_train):
    to_sensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("./datas", is_train, transform = to_sensor, download = True)
    return DataLoader(data_set, batch_size = 15, shuffle=True)


def evaluate_accuracy(test_data, net, use_cnn):
    n_current = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            if (use_cnn):
                outputs = net.forward(x.view(-1, 1, 28, 28))
            else:
                outputs = net.forward(x.view(-1, 28*28))

            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_current += 1
                n_total += 1
    return n_current / n_total


def main(is_infer, use_cnn, use_pic_predict):
    train_data = Get_Download_data(is_train = True)
    test_data = Get_Download_data(is_train = False)

    if (use_cnn):
        net = CNN_Net()
    else:
        net = FC_Net()

    if (is_infer) :
        print("model is trained , now starting inferencing!!!!!")
        state_dic = torch.load('max_params.pth', weights_only= True)
        net.load_state_dict(state_dic)
        net.eval()
        test = True
        if (use_pic_predict):
            image = Image.open("self_test_pics/strange8.jpg").convert('L')
            image = np.array(image)
            image = 255 - image
            image = Image.fromarray(image)
            trans = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            if (test):
                root = tk.Tk()
                root.title("手写数字识别")
                canvas = tk.Canvas(root, width = 280, height = 280, bg = 'white')
                canvas.pack()

                pic = Image.new("L", (28, 28), 255)
                draw = ImageDraw.Draw(pic)

                def paint(event):
                    x1, y1 = (event.x - 1), (event.y - 1)
                    x2, y2 = (event.x + 1), (event.y + 1)
                    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
                    draw.line([x1, y1, x2, y2], fill="black", width=10)

                def clear_canvas():
                    canvas.delete("all")
                    draw.rectangle((0, 0, 280, 280), fill="white")

                def recognize_digit():
                    # # 调整图像大小为 28x28
                    # resized_image = pic.resize((28, 28))
                    # # 转换为 Tensor 并进行预处理
                    # image_tensor = trans(resized_image).unsqueeze(0)
                    # # 进行预测
                    # with torch.no_grad():
                    #     output = net(image_tensor)
                    #     predicted_class = torch.argmax(output).item()

                    # # 显示预测结果
                    # result_label.config(text=f"预测结果：{predicted_class}")
                    # 获取画布内容的 PostScript 表示
                    ps = canvas.postscript(colormode='mono')

                    # 将 PostScript 转换为 Pillow Image 对象
                    image = Image.open(io.BytesIO(ps.encode('utf-8'))).convert('L')

                    # 转换为 Tensor 并进行预处理
                    image_tensor = trans(image).unsqueeze(0)

                    # 展平图像张量
                    image_tensor = image_tensor.view(image_tensor.size(0), -1)

                    # 进行预测
                    with torch.no_grad():
                        output = net(image_tensor)
                        predicted_class = torch.argmax(output).item()

                    # 显示预测结果
                    result_label.config(text=f"预测结果：{predicted_class}")   
                    print(f"predict result :{predicted_class}") 

                canvas.bind("<B1-Motion>", paint)
                clear_button = tk.Button(root, text="清除", command=clear_canvas)
                clear_button.pack()
                recognize_button = tk.Button(root, text="识别", command=recognize_digit)
                recognize_button.pack()

                result_label = tk.Label(root, text="预测结果：") 
                result_label.pack()
                root.mainloop()


            if (test == False):
                image_tensor = trans(image).unsqueeze(0)

                image_tensor = image_tensor.view(image_tensor.size(0), -1)

                plt.imshow(image_tensor.squeeze().numpy().reshape(28,28), cmap='gray')
                plt.title("Transformed Image")
                plt.show()

                with torch.no_grad():
                    out = net(image_tensor)
                    result = torch.argmax(out).item()

                print("prediction_result is :", result)

        else:
            print("now max params inference accuracy is : ", evaluate_accuracy(test_data, net, use_cnn))
    else:


        print("initial accuracy : ", evaluate_accuracy(test_data, net, use_cnn))

        optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)


        max_accyracy = -1
        start_time = time.time()
        for epoch in range (8):
                for (x, y) in train_data:
                    net.zero_grad()
                    if (use_cnn):
                        output = net.forward(x.view(-1, 1, 28, 28))
                    else:
                        output = net.forward(x.view(-1, 28*28))

                    loss = torch.nn.functional.nll_loss(output, y)
                    loss.backward()
                    optimizer.step()

                end_time = time.time()

                now_accuracy = evaluate_accuracy(test_data, net, use_cnn)
                if (now_accuracy > max_accyracy):
                    max_accyracy = now_accuracy
                    torch.save(net.state_dict(), 'max_params.pth')
                    print("max_params's epoch : ", epoch)
                
                
                print ("epoch :", epoch, "now_accyracy: ", now_accuracy, "max_accuracy", max_accyracy)
                
                if (use_cnn):
                    print ("use cnn opoch  ", epoch, "time consume:", end_time - start_time)
                else:
                    print ("use fc opoch  ", epoch, "time consume:", end_time - start_time)
            
            
                



    # for (n, (x, _)) in enumerate(test_data):
    #     if n > 5:
    #         break
    #     predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
    #     plt.figure(n)
    #     plt.imshow(x[0].view(28*28))
    #     plt.title("prediction : " + str(int(predict)))

    # plt.show()

    for (n, (x, _)) in enumerate(test_data):
        if n > 5:
            break

        if (use_cnn):
            predict = torch.argmax(net.forward(x[0].view(-1, 1,28, 28)))
        else:
            predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))

        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap='grey')
        plt.title("prediction: " + str(int(predict)))
    plt.show()
    # os.system(["pause"])


if __name__ == "__main__":

    is_infer = False
    use_cnn = False
    use_pic_predict = False
    if (len(sys.argv) < 4):
        print("1.error, correct run this : python first_try.py True/False ")
        print("2. please confirm if use cnn")
        print("3.confirm if you use your own picture to predict")
        sys.exit(1)
    
    if (sys.argv[1] == "False"):
        is_infer = False
    else:
        is_infer = True

    if (sys.argv[2]) == "use_cnn":
        use_cnn = True
    else:
        use_cnn = False

    if (sys.argv[3] == "test_own_pic"):
        use_pic_predict = True
    else:
        use_pic_predict = False


    main(is_infer, use_cnn, use_pic_predict)

                

