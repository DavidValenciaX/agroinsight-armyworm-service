from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import asyncio
from datetime import datetime
import uuid
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

# Cargar el modelo al iniciar la aplicación
def load_model():
    try:
        # Verificar que el archivo existe
        import os
        model_path = 'pesos_modelo_identificacion_gusano_cogollero.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encuentra el archivo de pesos: {model_path}")
            
        # Verificar el tamaño del archivo
        file_size = os.path.getsize(model_path)
        print(f"Tamaño del archivo de pesos: {file_size} bytes")
        
        # Intentar cargar el modelo
        state_dict = torch.load(model_path, map_location=device)
        
        # Verificar que el state_dict tiene las claves esperadas
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        if expected_keys != loaded_keys:
            print("Advertencia: Las claves del modelo no coinciden")
            print("Claves faltantes:", expected_keys - loaded_keys)
            print("Claves extras:", loaded_keys - expected_keys)
            
        model.load_state_dict(state_dict)
        print("Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise

# Cargar el modelo
model = load_model()
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

async def process_single_image(image_data, filename: str):
    try:
        # Validar y abrir la imagen
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
            "filename": filename,
            "status": "success",
            "predicted_class": CLASSES[predicted_class],
            "confidence": probabilities[predicted_class],
            "probabilities": {
                class_name: prob 
                for class_name, prob in zip(CLASSES, probabilities)
            }
        }
        
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "error": str(e)
        }

@app.post("/multi-predict")
async def predict_multiple(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se han proporcionado archivos"
        )
    
    try:
        results = []
        tasks = []
        
        # Crear tareas asíncronas para procesar cada imagen
        for file in files:
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "El archivo debe ser una imagen"
                })
                continue
                
            image_data = await file.read()
            task = asyncio.create_task(process_single_image(image_data, file.filename))
            tasks.append(task)
        
        # Esperar a que todas las tareas se completen
        if tasks:
            processed_results = await asyncio.gather(*tasks)
            results.extend(processed_results)
        
        # Analizar resultados para determinar el código de estado
        successful_predictions = [r for r in results if r["status"] == "success"]
        failed_predictions = [r for r in results if r["status"] == "error"]
        
        if not results:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "message": "No se pudo procesar ninguna imagen",
                    "results": []
                }
            )
        
        if len(successful_predictions) == len(results):
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "Todas las imágenes fueron procesadas exitosamente",
                    "results": results
                }
            )
        
        if len(failed_predictions) == len(results):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "message": "No se pudo procesar ninguna imagen",
                    "results": results
                }
            )
        
        return JSONResponse(
            status_code=status.HTTP_207_MULTI_STATUS,
            content={
                "message": "Algunas imágenes no pudieron ser procesadas",
                "successful": len(successful_predictions),
                "failed": len(failed_predictions),
                "results": results
            }
        )
            
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Error interno del servidor",
                "error": str(e)
            }
        )

@app.get("/")
async def root():
    return {
        "message": "API de Detección de Gusano Cogollero",
        "endpoints": {
            "/predict": "POST - Envía una imagen para analizar",
            "/multi-predict": "POST - Envía un lote de imágenes para analizar",
            "/": "GET - Muestra esta información"
        }
    } 