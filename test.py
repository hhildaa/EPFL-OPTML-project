import torch

def test(model, test_loader, device):
    """
    Test the model on the test dataset.
    """

    num_correct = 0
    num_samples = 0

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        #Get predictions
        output = model(inputs)

        #Keep track of correct predictions and number of samples to calculate accuracy later
        _, predictions = torch.max(output.data, 1)
        num_correct += (predictions == labels).sum().item()
        num_samples += predictions.size(0)


    #Calculate epoch accuracy and average loss
    accuracy = float(num_correct) / float(num_samples) * 100
            
    print(f"Test accuracy {accuracy}")
    return accuracy
