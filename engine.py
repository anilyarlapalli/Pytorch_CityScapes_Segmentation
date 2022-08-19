from cgi import test
import dataset
import torch
import config
from tqdm import tqdm
import utils
import tqdm
from tqdm import notebook


def training(model, epochs, batch_size, train_iter, test_iter, optimizer, criterion):
    logs = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    main_pbar = tqdm.notebook.tqdm(range(epochs))
    main_pbar.set_description('common progress ')
    history = dict(train_loss = [], 
                   train_dice = [], 
                   test_loss = [], 
                   test_dice = [])

    
    for epoch in main_pbar:
        running_params = dict(train_loss = [], 
                               train_dice = [], 
                               test_loss = [], 
                               test_dice = [])
        train_pbar = tqdm.notebook.tqdm(range(len(train_iter)))
        
        for step in train_pbar:
             
            train_imgs, train_masks = next(iter(train_iter))
            train_imgs, train_masks = train_imgs.to(device), train_masks.to(device)
            
            optimizer.zero_grad()
            
            train_predictions = model(train_imgs)

            train_loss = criterion(train_predictions, train_masks)
            train_loss.backward()
            
            train_dice = utils.dice(pred = train_predictions, label = train_masks)
            
            optimizer.step()
        
        
       
            with torch.no_grad():
                test_images, test_masks = next(iter(test_iter))
                test_images, test_masks = test_images.to(device), test_masks.to(device)
            
                test_predictions = model(test_images)
    
                test_loss = criterion(test_predictions, test_masks)
        
                test_dice = utils.dice(pred = test_predictions, label = test_masks)
                
            
            current_metrics = dict(train_loss = [train_loss.item(), ], 
                                   train_dice = [train_dice.item(), ], 
                                   test_loss = [test_loss.item(),], 
                                   test_dice = [test_dice.item(),])
            
            running_params.update(current_metrics)
            
            mean_metrics = dict(zip(running_params.keys(), [(sum(tensor) / (step + 1)) for tensor in running_params.values()]))
    
            train_pbar.set_postfix(mean_metrics)
            torch.cuda.empty_cache()
        
        temp = [train_loss.item(), train_dice.item(), test_loss.item(), test_dice.item()]
        logs.append(temp)
        history.update(running_params)
        best_loss = max(history['test_loss'])
        best_loss_index =  history['test_loss'].index(best_loss)
        current_loss_index = history['test_loss'].index(test_loss.item())
        if abs(current_loss_index - best_loss_index) >= 5:
            for param_group in optimizer.param_groups:
                if param_group['lr'] * 0.1 > 1e-6:
                    print('reduce learning rate to', {param_group['lr'] * 0.1})
                    param_group['lr'] *= 0.1

    torch.save({ 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, "best.pt")

    return logs