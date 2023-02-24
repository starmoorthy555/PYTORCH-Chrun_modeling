import pandas as  pd # Import Pandas library

data = pd.read_csv("/home/soft25/Downloads/Churn_Modelling.csv") #Reading the dataset

x = data.iloc[:,3:13].values #Split the dataset into x and y
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder #Convert the catagerical value into numerical value
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])

from sklearn.model_selection import train_test_split #Split the data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler #Scalling the data
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from torch import nn #Import pytorch module
import torch.nn.functional as F #Import pytoch functional

class ANN(nn.Module): #Create a Class for ANN model
    def __init__(self,input_dim=10,output_dim=1): #Define init function
        super(ANN,self).__init__()
        self.input = nn.Linear(input_dim,64) #Initializing the input layer
        self.hidden1 = nn.Linear(64,32) #Initializing the hidden layer
        self.hidden2 = nn.Linear(32, 32) #Initializing the another hidden layer
        self.out = nn.Linear(32,1) #Initializing the output layer
        self.Dropout = nn.Dropout(0.2) #Initializing the Dropoutlayer for preventing overfitting
        
    def forward(self,x):
        x = F.relu(self.input(x)) #Initializing the activation function for first input layer
        x = F.relu(self.hidden1(x)) #Initializing the activation function for first hidden layer
        x = F.relu(self.hidden2(x)) #Initializing the activation function for secopnd hidden layer
        x = F.sigmoid(self.out(x)) #Initializing the activation function for output layer
        return (x)
    
model = ANN(input_dim=10,output_dim=1) #Gives the input for ouir model
print(model)

x_train.shape
y_train.shape


import torch #importing the torch
import torch.utils.data #import torch.utils.data library
from torch.autograd import Variable #Import Variable from pytorch

x_train = torch.from_numpy(x_train) #Convert the ndarry to tensor for x_train
y_train = torch.from_numpy(y_train).view(-1,1) #Convert the ndarray to tensor for y_train

x_test = torch.from_numpy(x_test) #Convert the ndarry to tensor for x_test
y_test = torch.from_numpy(y_test).view(-1,1) #Convert the ndarry to tensor for y_test

train = torch.utils.data.TensorDataset(x_train,y_train) #Convert the python numerical value into pytorch tensor for train data
test = torch.utils.data.TensorDataset(x_test,y_test) #Convert the python numerical value into pytorch tensor for test data

train_loader = torch.utils.data.DataLoader(train,batch_size=64) #Iterate through the data and manage batches using DataLoader
test_loader = torch.utils.data.DataLoader(test,batch_size=64) #Iterate through the data and managhe the batchsize using DataLoader

import torch.optim as optim #Import Optimization
loss_fn  = nn.BCELoss() #Declare the Loss_function
optimizer = optim.SGD(model.parameters(),lr=0.01) #Declare the optimization

epochs = 35 #Initialize the Epoch value

epoch_list=[] #List for stor the epoch value
train_loss_list =[] #List for stor the loss value from the training data
val_loss_list = [] #List for store the loss value for the val data
train_acc_list=[] #List for store the accuracy for training data
val_acc_list=[] #List for store the accuracy for validation data

model.train() #Create a model for training

for epoch in range(epochs): #Model for train
    train_loss = 0.0 #Detrmine the train_loss value as zero
    val_loss = 0.0 #Determine the val_loss as zero
    
    correct = 0 #Determione the correct value as zero
    total = 0 #Determine the total value as zero
    
    for data,target in train_loader: #Split the data inti data variable and target variable.
        data = Variable(data).float() #datavariable for training data
        target = Variable(target).type(torch.FloatTensor) #Target variable for predicting

        optimizer.zero_grad() #Optimizer for Restart looping without losses
        output =model(data) #Output with the help of model
        predicted = (torch.round(output.data[0])) #predict the output
        # print(target)
        total += len(target) #Length of the predict value
        correct += (predicted==target).sum()
        
        loss = loss_fn(output,target)#Calculate the loss vale using output and target value.
        
        loss.backward()#Reverse direction to comput the gradients and accumulate their values
        
        optimizer.step()#Optimzer the loss value and update the weights
        
        train_loss += loss.item()*data.size(0)#Update traaining loss
        
    train_loss = train_loss/len(train_loader.dataset)#Calculate the average training loss 

    accuracy = 100*correct/float(total)#Calculate  the accuracy
    
    train_acc_list.append(accuracy)#Append the accuracy value in the accuracy list
    train_loss_list.append(train_loss)#Append the train loss value in that paticular list
    
    print('epoch : ',epoch+1,'accuracy : ',accuracy,'loss : ',train_loss)

epoch_list.append(epoch +1)

print(epoch_list)

#Preparing for testing dataset
data1 = pd.read_csv("/home/soft25/Downloads/Churn_Modelling.csv") #Reading the Dataset

from sklearn.model_selection import train_test_split #import train test split 
train,test = train_test_split(data1,test_size =0.25,random_state=0) #Covert the train and test data set
test = test.iloc[:,3:13].values 
from sklearn.preprocessing import LabelEncoder #Convert the categerical variable into numeri
le = LabelEncoder()
test[:,1] = le.fit_transform(test[:,1])
test[:,2] =  le.fit_transform(test[:,2])
from sklearn.preprocessing import StandardScaler#Normalizing the test dataset
sc = StandardScaler()
test = sc.fit_transform(test)
test1 = torch.from_numpy(test)#Convert the numpy data into tensor
test3 = Variable(test1).float()#Convert the data tensor into variable

results = [] #Make a list for store the output value
model.eval() #Make a model evaluation
#with torch.no_grab:
for data in test3: 
    output = model(data) #Predict the output using the model
    pred = int(torch.round(output.data[0]).item()) #make the output value in round
    results.append(pred) #Append the output value

print(results)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(results, y_test) #Make a confusion metrix for our output value and the y_test value
cr = classification_report(results, y_test) #Make a classification report for output value and y_test value
print(cm) #print the confusion metrix
print(cr) #Print the classification report
