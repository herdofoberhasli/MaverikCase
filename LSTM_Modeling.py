#Long Short Term Memory Neural Net - Dayton

#Create function that does the whole neural net
def neural_net(dataset,name = '',batch_size = 16, nn_shape = [1,4,1], learning_rate = .001, num_epochs = 10, train_amount = .95):
    '''
    - dataset is a csv file that contain the different data that we want run. The data provided
    should already be in a useable form. For the somewhat lengthy way of prepping this data, ask Dayton
    for the functions he built.
    - name is a temporary hold waiting for an elegant solution. It is the name of the dataset.
    - batch_size is the size of the batch for training
    - nn_shape is number of input nodes, hidden nodes, and then number of stacked layers.
    - learning_rate is the learning rate of the neural net
    - num_epochs is how many epochs we want to run through in the training.
    
    Returns: The RMSE of each model
    
    ***Huge thank you to Greg Hogg for the help in structuring the set up of the pytorch nn***

    '''
    #Getting things put together in PyTorch for the test. (Big thanks to Greg Hogg)

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import random
    from sklearn.preprocessing import MinMaxScaler
    from copy import deepcopy as dc
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import mean_absolute_percentage_error
    
    #Select device. This is essentially looking for a usuable gpu. (Useful in Google colab)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    #Create original of the dataset for use later
    og_dataset = dataset
    
    #Remove site 23065 as per class notes
    og_dataset = og_dataset.drop(og_dataset[og_dataset.site_id_msba == 23065].index)
    dataset = dataset.drop(og_dataset[dataset.site_id_msba == 23065].index)
    
    
    #Get list of stores and the number of stores.
    stores_list = dataset['site_id_msba'].unique().tolist()
    num_stores = len(stores_list)
    num_holdout = num_stores - math.floor(.95*num_stores)
    
    #Randomly choose {round up of .95 num stores} stores from the list as holdout stores.
    #holdout_stores = random.sample(stores_list,num_holdout)
    #train_stores = [store for store in stores_list if store not in holdout_stores]
    
    #Use specific stores
    holdout_stores = [24220,23555]
    train_stores = [store for store in stores_list if store not in holdout_stores]
    
    train_data_idx = dataset.index[dataset['site_id_msba'].isin(train_stores)]
    test_data_idx = dataset.index[dataset['site_id_msba'].isin(holdout_stores)]
    
    #Get important split numbers
    split_idx = len(train_data_idx)
    
    #Spereate into train and test for rearrange
    train_data = dataset.iloc[train_data_idx]
    test_data = dataset.iloc[test_data_idx]
    
    dataset = pd.concat([train_data, test_data], axis=0)
    
    #Remove columns not needed for training
    dataset = dataset.drop(columns=['site_id_msba', 'calendar.calendar_day_date','capital_projects.soft_opening_date', 'calendar.day_of_week'])
    
    #Getting the amount of predictors
    col = int(len(dataset.columns)) - 1
    
    #Scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    
    #Shift to numpy not needed as it is already
    
    #Seperate into train and test
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    
    #Create y_train and X_train with non-holdout
    y_train = train_data[:, 0]
    X_train = train_data[:,1:]
    
    #Create y_test_ and X_test with holdout
    y_test = test_data[:, 0]
    X_test = test_data[:, 1:]
    
    
    
    '''The following split is to be used if not using specifc stores as the holdout
    #Adjust df to numpy
    shifted_df_as_np = dataset.to_numpy()
    
    #Scale everything
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    
    
    #Separate into X and y.
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    
    
    #Split into train and test
    split_index = int(len(X) * train_amount)
    
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]'''
    
    #Adding additional dimension needed for pytorch
    X_train = X_train.reshape((-1, col, 1))
    X_test = X_test.reshape((-1, col, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    
    #Change vars to tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    
    #Set up dataset as time series
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    #Set up dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #Put the batches on the device
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break
    
    #Create LSTM class
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                                batch_first=True)
            
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    #Create model and put on device
    model = LSTM(nn_shape[0], nn_shape[1], nn_shape[2])
    model.to(device)
    
    #Function for training one epoch
    def train_one_epoch():
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()
    
    #Function for validating one epoch
    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0
        
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)
        
        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()
    
    #Set parameters
    learning_rate = learning_rate
    num_epochs = num_epochs
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #Do the training and validating
    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()
    
    #Get predictions
    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()
    
    #Seperate everything out
    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], col+1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)

    train_predictions = dc(dummies[:, 0])
    ##############
    dummies = np.zeros((X_train.shape[0], col+1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_train = dc(dummies[:, 0])
    ##############
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], col+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])
    ##############
    dummies = np.zeros((X_test.shape[0], col+1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_test = dc(dummies[:, 0])
    ##############
    
    
    #Create RMSE function
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    rmse = rmse(test_predictions, new_y_test)
    
    mape = mean_absolute_percentage_error(new_y_test, test_predictions)
    
    print(f"The RMSE of this dataset is {rmse}.")
    print(f"The MAPE of this dataset is {mape}.")
    
    #Plotting
    #Set up individical parameters
    if (name == "Unleaded") | (name == "Diesel"):
        units = 'gallons'
    else:
        units = 'dollars'
        
    #Make the plots. This will need to be generalized for using the random version and not set to two.
    plt.plot(new_y_test[:366], label='Sales')
    plt.plot(test_predictions[:366], label='Predicted Sales')
    plt.title(f'Store {holdout_stores[0]} ({name})')
    plt.xlabel('Day')
    plt.ylabel(f'Sales ({units})')
    plt.legend()
    plt.show()
    
    plt.plot(new_y_test[366:], label='Sales')
    plt.plot(test_predictions[366:], label='Predicted Sales')
    plt.title(f'Store {holdout_stores[1]} ({name})')
    plt.xlabel('Day')
    plt.ylabel(f'Sales ({units})')
    plt.legend()
    plt.show()
    return('Done.')

#%%
import pandas as pd
#Read in the data
file_path_unleaded = "~/Desktop/UtahMSBA/Capstone3/Data/unleaded.csv"
file_path_diesel = "~/Desktop/UtahMSBA/Capstone3/Data/diesel.csv"
file_path_indoor = "~/Desktop/UtahMSBA/Capstone3/Data/indoor.csv"
file_path_food = "~/Desktop/UtahMSBA/Capstone3/Data/food.csv"
def load_the_data(file_path_unleaded,file_path_diesel,file_path_indoor,file_path_food):
    unleaded_data = pd.DataFrame(pd.read_csv(file_path_unleaded))
    diesel_data = pd.DataFrame(pd.read_csv(file_path_diesel))
    indoor_data = pd.DataFrame(pd.read_csv(file_path_indoor))
    food_data = pd.DataFrame(pd.read_csv(file_path_food))
    return unleaded_data, diesel_data, indoor_data, food_data
#%%
unleaded, diesel, indoor, food = load_the_data(file_path_unleaded, file_path_diesel, file_path_indoor, file_path_food)

#%%
neural_net(unleaded, 'Unleaded')
neural_net(diesel, 'Diesel')
neural_net(indoor, 'Indoor')
neural_net(food,'Food')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    