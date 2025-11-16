import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class ConstructionTaskClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = models.vit_b_16(weights=None)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

class ConstructionTaskPredictor:
    def __init__(self, model_path, classes, device='cuda'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to saved model weights
            classes: List of class names (e.g., ['installing_led_lights', 'laying_bricks', ...])
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.classes = classes
        
        # Load model
        self.model = ConstructionTaskClassifier(len(classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Predict the task from an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with 'task' and 'confidence'
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        task = self.classes[predicted.item()]
        confidence = confidence.item()
        
        # Get top 3 predictions
        top3_prob, top3_idx = probabilities.topk(min(3, len(self.classes)), dim=1)
        top3_predictions = []
        for i in range(top3_idx.size(1)):
            top3_predictions.append({
                'task': self.classes[top3_idx[0, i].item()],
                'confidence': top3_prob[0, i].item()
            })
        
        return {
            'task': task,
            'confidence': confidence,
            'top3': top3_predictions
        }
    
    def predict_batch(self, image_paths):
        """Predict tasks for multiple images"""
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results