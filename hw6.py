import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    data_set=datasets.FashionMNIST('./data',train=training,
                                   download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size = 64)
    return loader
    
def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )

    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):  # loop over the dataset multiple times
        correct = 0
        total_loss = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_loss.append(loss.item())
            opt.step()
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)}({(100*correct/len(train_loader.dataset)):.2f}%) Loss: {sum(total_loss)/len(total_loss):.3f}")

    

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total_loss = []
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss.append(loss.item())
        acc = 100 * correct/10000
        if show_loss:
            print(f'Average loss: {sum(total_loss)/len(total_loss):.4f}\nAccuracy: {acc:.2f}%')
        else: 
            print(f'Accuracy: {acc:.2f}%')
    
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    logits = model(test_images)
    prob = F.softmax(logits, dim=1)

    i = torch.tensor([index])
    select = torch.index_select(prob, 0, i)
    sort, indices = torch.sort(select,descending=True)
    s = sort.tolist()[0]
    ID = indices.tolist()[0]
    print(f'{class_names[ID[0]]}: {s[0]*100:.2f}% \n{class_names[ID[1]]}: {s[1]*100:.2f}%\n{class_names[ID[2]]}: {s[2]*100:.2f}%')



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    print(type(test_loader))
    print(test_loader.dataset)
    model = build_model()
    print(model)
    criterion = nn.CrossEntropyLoss()
    T = 5
    train_model(model, train_loader, criterion, T)
    evaluate_model(model, test_loader, criterion, True)
    evaluate_model(model, test_loader, criterion, False)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)
