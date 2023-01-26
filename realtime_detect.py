import cv2 as cv
import torch
import numpy as np


def load_model(model, source : str = 'local' or 'github'):
    if source == 'github':
        model = torch.hub.load('ultralytics/yolov5', model, pretrained=True)
    elif source == 'local':
        model = torch.hub.load('D:\DEVELOPPEMENTS\MACHINE_LEARNING\yolov5', model, source='local', pretrained=True)
    return model


if __name__ == '__main__':
    
    # Load model
    model = load_model('yolov5s', source='local')
    
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        
        result = model(frame)
        
        cv.imshow('Yolo_realtime_detection', np.squeeze(result.render()))

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
