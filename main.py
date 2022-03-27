import numpy as np
import random
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
from torchvision import models
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def images_transforms(phase):
    if phase == 'train':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_transformation


class ResNet18(nn.Module):
   def __init__(self, num_class, pretrained_option=False):
        super(ResNet18,self).__init__()
        self.model=models.resnet18(pretrained=pretrained_option)
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons, num_class)
        
   def forward(self,X):
        out=self.model(X)
        return out


def training():
    model.to(device)
    best_model_wts = None
    best_evaluated_acc = 0
    train_acc = []
    test_acc = []
    test_Recall = []
    test_Precision = []
    test_F1_score = []
    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for _, (data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                        
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct / len(train_loader.dataset)) * 100.
            print ("Epoch : " , epoch)
            print ("Loss : " , total_loss)
            print ("Correct : " , correct)
        accuracy, Recall, Precision, F1_score = evaluate()
        train_acc.append(correct)  
        test_acc.append(accuracy)
        test_Recall.append(Recall)
        test_Precision.append(Precision)
        test_F1_score.append(F1_score)

        if accuracy > best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    #save model
    torch.save(best_model_wts, "weight.pt")
    model.load_state_dict(best_model_wts)

    return train_acc , test_acc , test_Recall , test_Precision , test_F1_score


def evaluate(plot=False):
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for _, (data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            for j in range(data.size()[0]):
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        print ("num_correct :", correct, " / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : ",  Recall)

        Precision = TP/(TP+FP)
        print ("Preecision : ",  Precision)

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : ", F1_score)

        correct = (correct / len(test_loader.dataset)) * 100.
        print ("Accuracy : ", correct ,"%")
        
        if plot == True:
            plot_confusion_matrix(TP, TN, FP, FN)

    return correct, Recall, Precision, F1_score


def plot_confusion_matrix(TP, TN, FP, FN):
    array = [[TN, FP], [FN, TP]]
    df_cm = pd.DataFrame(array, index = ["Actual Normal", "Actual Pneumonia"],
                  columns = ["Predicted Normal", "Predicted Pneumonia"])
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig("./figure/cm.jpg")


def plot_result(result, ylabel, title, filename):
    epoch_seq = [i for i in range(epochs)]
    plt.plot(epoch_seq, result)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.show()
    plt.close()


if __name__=="__main__":
    IMAGE_SIZE = (128,128)
    batch_size = 128
    learning_rate = 1e-4
    epochs=30
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)

    trainset=datasets.ImageFolder('chest_xray/train', transform=images_transforms('train'))
    testset=datasets.ImageFolder('chest_xray/test', transform=images_transforms('test'))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResNet18(2, True)
    class_weights = torch.FloatTensor([5.0, 1.0]).to(device)
    Loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    train_acc, test_acc, test_Recall, test_Precision, test_F1_score  = training()
    plot_result(train_acc, "acc", "Train Accuracy", "./figure/train_acc.jpg")
    plot_result(test_acc, "acc", "Test Accuracy", "./figure/test_acc.jpg")
    plot_result(test_F1_score, "f1-score", "Test F1-Score", "./figure/test_f1_score.jpg")
    model.load_state_dict(torch.load("weight.pt"))
    model = model.to(device)
    evaluate(plot=True)
    