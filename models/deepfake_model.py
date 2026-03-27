import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def _build_model(self):
        model = models.efficientnet_b4(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model.to(self.device)

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        real_prob = probs[0].item()
        fake_prob = probs[1].item()
        return {
            "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
            "fake_probability": round(fake_prob * 100, 1),
            "real_probability": round(real_prob * 100, 1),
            "confidence": round(max(fake_prob, real_prob) * 100, 1),
        }