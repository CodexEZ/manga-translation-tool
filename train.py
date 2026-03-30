from ultralytics import YOLO
def train():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data = r"text_block.v2-manga-dataset.yolov8\data.yaml",
        epochs = 100,
        imgsz = 640,
        device = 0,
        project = r'D:\Projects\Manga Translator AI Model',
    )

if __name__ == '__main__':
    train()