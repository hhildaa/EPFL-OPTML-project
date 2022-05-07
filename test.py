import torch

def test(model, test_loader):
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data in test_loader:
            images, targets = data[0].to(model.device), data[1].to(model.device)        
            scores = model(images)
            _, predictions = torch.max(scores.data, 1)
            num_correct += (predictions == targets).sum().item()
            num_samples += predictions.size(0)
            
    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
