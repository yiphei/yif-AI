import torch
import os

def model_fn(model_dir):
    """
    Load the PyTorch model from the specified directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assuming the model is saved as `model.pth` in the model directory
    model_path = os.path.join(model_dir, 'model.pth')
    
    model = torch.load(model_path, map_location=device)
    
    # Check if there are multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
        model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel

    model.to(device)
    
    # Load training data (assuming it's saved as 'training_data.pth' in the model directory)
    training_data_path = os.path.join(model_dir, 'training_data.pth')
    training_data = torch.load(training_data_path)

    return {"model": model, "training_data": training_data}


def predict_fn(input_data, model_and_data):
    """
    Make prediction on the input data using the loaded model.
    """
    model = model_and_data["model"]
    training_data = model_and_data["training_data"]


    # Data preparation
    with open(training_data, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    itoc = {i: c for i, c in enumerate(chars)}

    decoder = lambda x: "".join([itoc[i] for i in x])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Convert input data to tensor
        data = torch.tensor(input_data, device=device, dtype=torch.float32)
        
        # Get model predictions
        output = model(data)
        
        # Convert tensor to list for return
        result = output.cpu().numpy().tolist()
    
    return result
