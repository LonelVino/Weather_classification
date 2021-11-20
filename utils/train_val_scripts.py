import torch
import matplotlib.pyplot as plt
import math


# Simple function to plot images from batch
def show_batch(x, classes, nimgs=4, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]), denorm=True):
    if denorm: # denormalize image for plotting
      denorm_x = (x * std.view(1, 3, 1, 1)) + mean.view(1, 3, 1, 1)
    else: # Don't denormalize image, used for plotting images with single channel (Value channel for HSV image)
      denorm_x = x
    img = 0 # image counter
    nr = math.ceil(math.sqrt(nimgs)) # calculating number of rows and columns to plot
    fig, ax = plt.subplots(nrows=nr, ncols=nr, figsize=(15, 10))

    inner_break = False
    for i in range(nr):
        for j in range(nr):
            if denorm:
              ax[i][j].imshow(denorm_x[img].permute(1, 2, 0))
            else:
              ax[i][j].imshow(denorm_x[img][0])
            ax[i][j].set_title(classes[img])
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
            img += 1
            if img == nimgs:
                inner_break = True
                break

        if inner_break:
            break
            
            
            
def accuracy(out, y):
    """
    out: shape (batch_size x 2) logits
    y: shape (batch_size) class indices
    """
    bs = out.shape[0]
    out = torch.softmax(out, dim=1).argmax(dim=1).view(bs) # Applying softmax over prediction logits
    y = y.view(bs)
    corrects = (out == y).sum().float()
    acc = corrects/bs
    return acc


# Simple function to perform one Training epoch
def train_epoch(model, dl, criterion, optimizer, scheduler, device='cuda:0'):
    model.train() # putting model in training mode
    losses = [] # tracking running losses
    accuracies = [] # training running accuracy

    for x,y in dl:
        # Transferring batch to GPU
        x = x.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()

        out = model(x)
        bs_acc = accuracy(out, y) # Train accuracy for current batch
        loss = criterion(out, y) # Train loss for current batch
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item() * x.shape[0]) # running loss
        accuracies.append(bs_acc * x.shape[0]) # running accuracy

    epoch_loss = sum(losses) / len(dl.dataset) # epoch loss
    epoch_acc = sum(accuracies) / len(dl.dataset) # epoch accuracy
    return epoch_loss, epoch_acc


# Simple function to perform one Validation epoch
def valid_epoch(model, dl, criterion, device='cuda:0'):
    model.eval() # putting model in evaluation mode
    losses = [] # tracking running losses
    accuracies = [] # tracking running accuracy

    for x,y in dl:
        # Transferring batch to GPU
        x = x.to(device=device)
        y = y.to(device=device)

        # Using torch.no_grad() to prevent calculation of gradients hence saving memory as gradients are not required during validation phase
        with torch.no_grad():
            out = model(x)
            bs_acc = accuracy(out, y) # Validation accuracy for current batch
            loss = criterion(out, y) # Validation loss for current batch

        losses.append(loss.item() * x.shape[0]) # running loss
        accuracies.append(bs_acc * x.shape[0]) # running accuracy

    epoch_loss = sum(losses) / len(dl.dataset) # epoch loss
    epoch_acc = sum(accuracies) / len(dl.dataset) # epoch accuracy
    return epoch_loss, epoch_acc
