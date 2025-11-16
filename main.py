import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
from inference import ConstructionTaskPredictor
from training import ConstructionDataset, ConstructionTaskClassifier, train_model
import os


# ---- Training Script ----


if __name__ == '__main__':
    # Train model, if it has not been trained
    if not os.path.exists('best_construction_model.pth'):
        print("Training new model...")
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = ConstructionDataset('construction_dataset/train', transform=train_transform)
        val_dataset = ConstructionDataset('construction_dataset/val', transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Create model
        num_classes = len(train_dataset.classes)
        print(f'Training model for {num_classes} classes: {train_dataset.classes}')
        
        model = ConstructionTaskClassifier(num_classes)
        
        # Train
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        
        trained_model = train_model(model, train_loader, val_loader, num_epochs=10, device=device)
        
        print('Training complete!')



    # ---- Inference Script ----


    # Define your task classes (must match training order)
    classes = [
        'installing_led_lights',
        'laying_bricks',
        'pouring_concrete',
        # Add more tasks as needed
    ]
    
    # Initialize predictor
    predictor = ConstructionTaskPredictor(
        model_path='best_construction_model.pth',
        classes=classes,
        device='cuda'
    )
    
    # Batch prediction
    img_paths = [os.path.join('demo_dataset', img) for img in os.listdir('demo_dataset')]
    # batch_results = predictor.predict_batch(img_paths)
    # print("\n\nBatch Results:")
    # for img_path, result in zip(img_paths, batch_results):
    #     print(f"{img_path}: {result['task']} ({result['confidence']:.2%})")

    # Single image prediction
    test_image_path = img_paths[0]
    single_result = predictor.predict(test_image_path)
    print(f"\nSingle Image Result for {test_image_path}:")
    print(f"Predicted Task: {single_result['task']}")
    print(f"Confidence: {single_result['confidence']:.2%}")
