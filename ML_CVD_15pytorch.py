import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Load Data
df1 = pd.read_csv("dftnooutliers.csv")

# Recoding outcome variable to be binary
df1['cvdiahydd2'] = df1['cvdiahydd2'].map({1:1, 2:0, 3:0})

# Drop NAs
df1 = df1.dropna(subset=['cvdiahydd2', 'BMI', 'whval', 'cholval3', 'hdlval3', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk'])
print(len(df1.index))

# Pick X and y
X = df1[['BMI', 'whval', 'cholval3', 'hdlval3', 'omdiaval', 'omsysval', 'ommapval', 'ompulval', 'cigdyal', 'cotval', 'TotmWalWk', 'TotmSitWk', 'TotmModWk', 'TotmVigWk']] # choose predictors (X)
y = df1['cvdiahydd2'] # choose outcome (y)

# Split the dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# The groups (label y) are not equal in size, so resampling is required for training set 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
y_train = y_resampled # after upsampling, y train needs to be updated

# Scale the data to avoid weighting of features pre model
scaler = StandardScaler()
scaler.fit(X_resampled)
X_train = scaler.transform(X_resampled)
X_test = scaler.transform(X_test)

# Turn target/y data into np array
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
print(type(y_train)) #check whether that has worked
print(type(y_test))

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 64

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break

# Neural network implementation
# Define dimensions
input_dim = 14
output_dim = 1

# Set architecture
class CVDModel(nn.Module):
    def __init__(self, input_dim, output_dim): #define the layers first
        super(CVDModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): #then define the computation performed at each cell
        out = self.relu1(self.bn1(self.layer1(x)))
        out = self.relu2(self.bn2(self.layer2(out)))
        out = self.dropout1(out)
        out = self.relu3(self.bn3(self.layer3(out)))
        out = self.sigmoid(self.layer4(out))
        return out

# Check model
model1 = CVDModel(input_dim, output_dim)
print(model1)

# Define parameters for running the model
learning_rate = 0.001 #learing rate for gradient descent
loss_fn = nn.BCELoss() # binary crossentropy as loss
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=0.001) # Adam as optimiser with learning rate

# Train the model
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

# Number of epochs
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    train_loss, valid_loss = [], []

    # Training phase
    model1.train()
    for X, y in train_dataloader:
        # Move data to the appropriate device
        X, y = X.to(device), y.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward propagation
        output = model1(X)

        # Loss calculation
        loss = loss_fn(output, y.unsqueeze(-1))

        # Backward propagation
        loss.backward()

        # Weight optimization
        optimizer.step()

        # Append training loss
        train_loss.append(loss.item())
    
    # Initialize lists to store true labels and predictions for evaluation phase only
    all_true_labels = []
    all_predictions = []
    # Evaluation phase
    model1.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            # Move data to the appropriate device
            X, y = X.to(device), y.to(device)

            # Get model outputs
            output = model1(X)

            # Loss calculation
            loss = loss_fn(output, y.unsqueeze(-1))
            valid_loss.append(loss.item())

            # Convert probabilities to binary predictions for binary classification
            predicted = output.round()

            # Store true labels and predictions for the classification report
            all_true_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy for this epoch
    accuracy = (predicted == y).float().mean().item()

    # Print epoch results
    print(f"Epoch: {epoch}, Training Loss: {np.mean(train_loss):.4f}, Valid Loss: {np.mean(valid_loss):.4f}, Accuracy: {accuracy:.4f}")

# After all epochs, generate and print the classification report
class_report = classification_report(all_true_labels, all_predictions, target_names=['Class 0', 'Class 1'], zero_division=0)
print("Final Classification Report:\n", class_report)

# Examine confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
