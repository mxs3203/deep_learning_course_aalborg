import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

from PyTorchTutorial.DeepLearningCourse.Assignment2.NetworkModelConv import ConvNet

torch.manual_seed(0)

def compute_accuracy(y_pred, y_labels):
    y_pred = y_pred.cpu().data.numpy()
    y_pred = y_pred.argmax(axis=1)
    acc = y_pred - y_labels.cpu().data.numpy()
    return 100*np.sum(acc==0)/len(acc)


print('Preparing training data...')
# Training data.
X_train = pickle.load(open('../Data/X_train.p', 'rb'))
Y_train_oh = pickle.load(open('../Data/Y_train.p', 'rb'))
# Reshaping training data.
no_ts = X_train.shape[0] # No. of training examples.
no_tf = X_train.shape[1] # No. of time frames.
no_qb = X_train.shape[2] # No. of quefrency bins.
X_train = np.reshape(X_train,(no_ts, no_qb, no_tf))
# Data normalization.
uX = np.mean(X_train)
sX = np.std(X_train)
X_train = (X_train - uX) / sX
# From one-hot encoding to integers.
Y_train = np.zeros(no_ts)
for i in range(no_ts):
    Y_train[i] = np.where(Y_train_oh[i]==1)[0][0]
# To PyTorch tensors.
X_train = torch.cuda.FloatTensor(X_train)
Y_train = torch.cuda.LongTensor(Y_train)
train_data = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_data, batch_size=1024, shuffle=True)
# Validation data.
X_valid = pickle.load(open('../Data/X_valid.p', 'rb'))
Y_valid_oh = pickle.load(open('../Data/Y_valid.p', 'rb'))
no_vs = X_valid.shape[0] # No. of validation examples.
X_valid = np.reshape(X_valid,(no_vs,no_qb, no_tf))
X_valid = (X_valid - uX) / sX
Y_valid = np.zeros(no_vs)
for i in range(no_vs):
    Y_valid[i] = np.where(Y_valid_oh[i]==1)[0][0]
X_valid = torch.cuda.FloatTensor(X_valid)
Y_valid = torch.cuda.LongTensor(Y_valid)
# Test data.
X_test = pickle.load(open('../Data/X_test.p', 'rb'))
Y_test_oh = pickle.load(open('../Data/Y_test.p', 'rb'))
no_es = X_test.shape[0] # No. of test examples.

X_test = np.reshape(X_test,(no_es,  no_qb, no_tf))
X_test = (X_test - uX) / sX
Y_test = np.zeros(no_es)
for i in range(no_es):
    Y_test[i] = np.where(Y_test_oh[i]==1)[0][0]
X_test = torch.cuda.FloatTensor(X_test)
Y_test = torch.cuda.FloatTensor(Y_test)
no_cl = 3 # No. of classes.



################### Training ############################
model = ConvNet(conv1_size=32,conv2_size=16, pool_size=2, kernel_size=5)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters())
summary(model, (1,40,101))

print('Training the model...')
model.train()
no_epoch = 50 # No. of training epochs.
train_loss = []
val_loss = []
train_acc = []
val_acc = []
for epoch in range(no_epoch):
    # Mini-batch processing.
    mi = 1
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch.unsqueeze(dim=1)   ) # Forward pass.
        loss = criterion(y_pred.squeeze(), y_batch)

        # For validation.
        y_pred_val = model(X_valid.unsqueeze(dim=1)   )
        loss_val = criterion(y_pred_val.squeeze(), Y_valid)
        print('Epoch {}, Batch: {}, Train loss: {}, Validation loss: {}'.format(epoch+1, mi, loss.item(), loss_val.item()))
        loss.backward() # Backward pass.
        optimizer.step()
        train_loss.append(loss.cpu().data.numpy())
        val_loss.append(loss_val.cpu().data.numpy())
        train_acc.append(compute_accuracy(y_pred, y_batch))
        val_acc.append(compute_accuracy(y_pred_val, Y_valid))
        mi += 1
# We plot the loss curves.
fig, ax = plt.subplots()
ax.plot(train_loss, label='Train loss')
ax.plot(val_loss, 'r', label='Validation loss')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig("Plots/loss_sgd.png")
plt.show()
# We plot the accuracy curves.
fig2, ax2 = plt.subplots()
ax2.plot(train_acc, label='Train accuracy')
ax2.plot(val_acc, 'r', label='Validation accuracy')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
plt.savefig("Plots/acc_sgd.png")
plt.show()

# ---------------- #
# MODEL EVALUATION #
# ---------------- #
model.eval()
y_pred = model(X_test.unsqueeze(dim=1))
acc = compute_accuracy(y_pred.squeeze(), Y_test)
print('Test accuracy: ' + str(acc) + '%')

