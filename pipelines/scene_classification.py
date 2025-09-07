import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(num_classes=365)
checkpoint = torch.hub.load('places365/places365', 'resnet18_places365')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

home_classes = ['living_room', 'bedroom', 'kitchen']
shop_classes = ['supermarket', 'store', 'bookstore']
office_classes = ['office', 'conference_room']

def classify_environment_places365(frame):
    img = Image.fromarray(frame[..., ::-1])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, idx = torch.max(outputs, 1)
    predicted_class = idx.item()
    
    if predicted_class in home_classes:
        return "Home"
    elif predicted_class in shop_classes:
        return "Shop"
    elif predicted_class in office_classes:
        return "Office"
    else:
        return "Unknown"