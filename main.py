import cv2
from ultralytics import YOLO
import numpy as np

class YOLOProcessor:
    def __init__(self):
        self.models = {
            '1': {'name': 'Detección de objetos', 'model_path': 'yolo11l.pt'},
            '2': {'name': 'Segmentación', 'model_path': 'yolo11l-seg.pt'},
            '3': {'name': 'Puntos clave', 'model_path': 'yolo11l-pose.pt'},
            '4': {'name': 'Clasificación', 'model_path': 'yolo11l-cls.pt'}
        }
        self.current_model = None

    def load_model(self, option):
        if option in self.models:
            model_info = self.models[option]
            print(f"Cargando modelo para {model_info['name']}...")
            self.current_model = YOLO(model_info['model_path'])
            return True
        return False

    def process_frame(self, frame, option):
        if self.current_model is None:
            return frame

        results = self.current_model(frame)
        
        # Procesar resultados según el tipo de modelo
        if option == '1':  # Detección de objetos
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{results[0].names[cls]} {conf:.2f}', 
                              (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif option == '2':  # Segmentación
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    for mask in masks:
                        mask = mask.astype(np.uint8) * 255
                        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                        frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        elif option == '3':  # Puntos clave (pose)
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()  # Obtener los keypoints como numpy array
                    # Dibujar los keypoints
                    for kp in keypoints:
                        x, y, conf = kp
                        if conf > 0.5:  # Solo dibujar puntos con confianza mayor a 0.5
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 2)
                    
                    # Dibujar las conexiones entre puntos (esqueleto)
                    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
                              [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
                              [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], 
                              [3, 5], [4, 6]]
                    
                    for link in skeleton:
                        pt1, pt2 = link
                        if keypoints[pt1][2] > 0.5 and keypoints[pt2][2] > 0.5:
                            pt1_coords = (int(keypoints[pt1][0]), int(keypoints[pt1][1]))
                            pt2_coords = (int(keypoints[pt2][0]), int(keypoints[pt2][1]))
                            cv2.line(frame, pt1_coords, pt2_coords, (0, 255, 0), 2)

        elif option == '4':  # Clasificación
            for result in results:
                if hasattr(result, 'probs'):
                    cls = result.probs.top1
                    conf = result.probs.top1conf
                    cv2.putText(frame, f'Clase: {results[0].names[cls]} ({conf:.2f})', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def run_camera(self, option):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame")
                break

            processed_frame = self.process_frame(frame, option)
            cv2.imshow('YOLO Processing', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    processor = YOLOProcessor()
    
    while True:
        print("\nSeleccione el tipo de procesamiento:")
        print("1. Detección de objetos")
        print("2. Segmentación")
        print("3. Puntos clave (pose)")
        print("4. Clasificación")
        print("5. Salir")
        
        option = input("Ingrese su opción (1-5): ")
        
        if option == '5':
            break
        
        if option in ['1', '2', '3', '4']:
            if processor.load_model(option):
                processor.run_camera(option)
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()