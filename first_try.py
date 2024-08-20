import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import sys
# import numpy as np

# xp = np.arange(0, 5, 0.1)
# yp = np.sin(xp)
# plt.plot(xp, yp)
# plt.show()

class Net(torch.nn.Module) :
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.softmax(self.fc4(x), dim = 1)

        return x
    

def Get_Download_data(is_train):
    to_sensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("./datas", is_train, transform = to_sensor, download = True)
    return DataLoader(data_set, batch_size = 15, shuffle=True)


def evaluate_accuracy(test_data, net):
    n_current = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_current += 1
                n_total += 1
    return n_current / n_total


def main(is_infer):
    train_data = Get_Download_data(is_train = True)
    test_data = Get_Download_data(is_train = False)
    net = Net()
    if (is_infer) :
        print("model is trained , now starting inferencing!!!!!")
        state_dic = torch.load('max_params.pth', weights_only= True)
        net.load_state_dict(state_dic)
        net.eval()
        print("now max params inference accuracy is : ", evaluate_accuracy(test_data, net))
    else:


        print("initial accuracy : ", evaluate_accuracy(test_data, net))

        optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)


        max_accyracy = -1
        for epoch in range (10):
                for (x, y) in train_data:
                    net.zero_grad()
                    output = net.forward(x.view(-1, 28*28))
                    loss = torch.nn.functional.nll_loss(output, y)
                    loss.backward()
                    optimizer.step()

                now_accuracy = evaluate_accuracy(test_data, net)
                if (now_accuracy > max_accyracy):
                    max_accyracy = now_accuracy
                    torch.save(net.state_dict(), 'max_params.pth')
                    print("max_params's epoch : ", epoch)
                
                
                print ("epoch :", epoch, "now_accyracy: ", now_accuracy, "max_accuracy", max_accyracy)
            
            
                



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
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":

    is_infer = False
    if (len(sys.argv) < 2):
        print("error, correct run this : python first_try.py True/False")
        sys.exit(1)
    
    if (sys.argv[1] == "False"):
        is_infer = False
    else:
        is_infer = True


    main(is_infer)

                

