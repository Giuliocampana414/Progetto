from ultralytics import YOLO


def train_model():
    # Load your model
    model = YOLO("yolov8n-obb.pt") 

    # Run the training
    model.train(
        data="config.yaml", 
        epochs=10,  
        batch=4,        # Lower this to 4 to be safe for 6GB VRAM
        workers=0,      # SET THIS TO 0 to fix the WinError 1920 / Access error
    )

if __name__ == '__main__':
    train_model()

 