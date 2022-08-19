import dataset
import torch
import config
from tqdm import tqdm
import utils
import model
from torchviz import make_dot
import engine
import os
import config

data_train = dataset.CityScapesSegDataset(config.TRAIN_FILES, config.CROP_SIZE, config.TRAIN_DIR)
data_test = dataset.CityScapesSegDataset(config.TEST_FILES, config.CROP_SIZE, config.TEST_DIR)
batch_size = 16
train_iter = torch.utils.data.DataLoader(data_train, config.BATCH_SIZE, shuffle=True, drop_last=True,)
test_iter = torch.utils.data.DataLoader(data_test, config.BATCH_SIZE, shuffle=False, drop_last=True,)




histories = []   

for model_name, model_instance in model.models_dict.items():
    name = model_name
    model_name = model_instance
    model_name.to("cuda:0")
    x = torch.zeros(8, 3, 128, 128, dtype=torch.float, requires_grad=False)
    x = x.to("cuda:0")
    outputs_x = model_name(x)
    make_dot(outputs_x, params=dict(list(model_name.named_parameters())))
    
    optimizer = torch.optim.Adam(params = model_name.parameters(), 
                             lr=1e-4, 
                             betas=(0.9, 0.999), 
                             eps=1e-08, 
                             weight_decay=0, 
                             amsgrad=False)

    criterion = torch.nn.BCEWithLogitsLoss()
    

    logs = engine.training(model_name,config.N_EPOCHS, config.BATCH_SIZE, train_iter, test_iter, optimizer, criterion)

    histories.append(logs)
    
    # Saving the model weights for each model in working directory
    torch.save({ 
    'model_state_dict': model_name.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, f"models/{model_name}_best.pt")
