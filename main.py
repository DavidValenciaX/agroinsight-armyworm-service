from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import timm

app = FastAPI(
    title="Gusano Cogollero API",
    description="API para detectar el estado de hojas de maíz afectadas por el gusano cogollero",
    version="1.0.0"
)

# Definir las clases
CLASSES = ['damaged_leaf', 'healthy_leaf', 'leaf_with_larva']

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD)
])

def preprocess_image(image):
    # Aplicar transformaciones
    img_tensor = transform(image)
    # Agregar dimensión de batch
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# Cargar el modelo al iniciar la aplicación
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recrear el modelo
model_name = 'caformer_s18.sail_in22k_ft_in1k_384'
model = timm.create_model(model_name, pretrained=False)

# Configurar el modelo para que coincida con el número de clases
num_classes = len(CLASSES)
in_features = model.head.fc.fc1.in_features

# Crear una nueva cabeza más simple manteniendo las capas necesarias
class NewHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.global_pool = model.head.global_pool
        self.norm = model.head.norm
        self.flatten = model.head.flatten
        self.drop = model.head.drop
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x

model.head = NewHead(in_features, num_classes)

# Cargar el diccionario de estado
model.load_state_dict(torch.load('pesos_modelo_identificacion_gusano_cogollero.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer y validar la imagen
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "El archivo debe ser una imagen"}
            )
        
        # Leer la imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocesar la imagen
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(device)
        
        # Realizar la predicción
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            probabilities = probabilities[0].tolist()
        
        return {
            "predicted_class": CLASSES[predicted_class],
            "confidence": probabilities[predicted_class],
            "probabilities": {
                class_name: prob 
                for class_name, prob in zip(CLASSES, probabilities)
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al procesar la imagen: {str(e)}"}
        )

@app.get("/")
async def root():
    return {
        "message": "API de Detección de Gusano Cogollero",
        "endpoints": {
            "/predict": "POST - Envía una imagen para analizar",
            "/": "GET - Muestra esta información"
        }
    } 