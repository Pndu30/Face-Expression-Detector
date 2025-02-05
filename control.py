import torch
import time
import torch.nn.functional as F
from torchvision import transforms

def prep_model(model, PATH='./models/weights/weights.pth'):
    model = model()
    model.load_state_dict(torch.load(PATH, weights_only=True))    
    model.eval()
    return model
    

def inference(model, img):
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    transform = transforms.Compose([
            transforms.Resize((48, 48)),  # Assuming the images are 48x48 pixels
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel images (RGB)
            transforms.ToTensor(),
        ])
    img = transform(img)
    img = img.unsqueeze(0)

    logits = model(img)
    out = F.softmax(logits, dim=1)
    idx = torch.argmax(out, dim=1)
    cur = time.time()
    return labels[idx.item()], out[0, idx].item(), cur
