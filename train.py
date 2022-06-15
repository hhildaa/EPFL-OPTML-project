import torch
import torch.nn as nn
import torch.optim as optim
import time

def train(model, optimizer, criterion, train_loader, device):
    """Optimizing the trained model, with the specified optimizer using the criterion as loss. 
    """
    epoch = 1
    start_time = time.time()
    #Keep track of epoch accuracies and losses
    accuracies = []
    losses = []
    #Train until 100% accuracy
    while True:
        running_loss = 0.0
        num_correct = 0
        num_samples = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            #Get predictions
            output = model(inputs)
            loss = criterion(output, labels)
            #Optimise
            loss.backward()
            optimizer.step()

            #Keep track of correct predictions and number of samples to calculate accuracy later
            _, predictions = torch.max(output.data, 1)
            num_correct += (predictions == labels).sum().item()
            num_samples += predictions.size(0)

            #Update current epoch loss
            running_loss += loss.item()

        #Calculate epoch accuracy and average loss
        accuracy = float(num_correct) / float(num_samples) * 100
        accuracies.append(accuracy)
        losses.append(running_loss/len(train_loader))
        
        print(f"Epoch {epoch} accuracy: {accuracy:.2f} Loss: {running_loss/len(train_loader):.3f}")

        #Finish training once accuracy is 100% or loss is 0
        if accuracy == 100:
            print('Finished Training of AlexNet')
            print(f"Number of epochs until 100% accuracy: {epoch}")
            print(f"Time taken: {time.time() - start_time}")
            break
        epoch += 1
    return accuracies, losses, model