import cv2
import numpy as np
import time
import os
import sys  # Add this import
import tkinter as tk
from tkinter import ttk, messagebox
from cv2 import dnn
from math import ceil

class EmotionDetector:
    def __init__(self):
        # Get the directory where the executable/script is located
        self.base_path = self._get_base_path()
        
        # Define model paths
        self.emotion_model_path = os.path.join(self.base_path, 'models', 'emotion-ferplus-8.onnx')
        self.face_model_path = os.path.join(self.base_path, 'models', 'RFB-320', 'RFB-320.caffemodel')
        self.face_proto_path = os.path.join(self.base_path, 'models', 'RFB-320', 'RFB-320.prototxt')
        
        # Initialize models
        self.emotion_dict = {
            0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
            4: 'anger', 5: 'disgust', 6: 'fear'
        }
        
        # Model parameters
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.input_size = [320, 240]
        self.threshold = 0.5
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        
        # Initialize models
        self._init_models()
        
    def _get_base_path(self):
        """Get base path for resources that works both in dev and PyInstaller"""
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            return os.path.join(sys._MEIPASS, '')  # Added empty join to ensure proper path format
        else:
            # Running in normal Python environment
            return os.path.dirname(os.path.abspath(__file__))
            
    def _init_models(self):
        """Initialize the face detection and emotion recognition models"""
        try:
            print(f"Loading emotion model from: {self.emotion_model_path}")
            print(f"Loading face model from: {self.face_model_path}")
            print(f"Loading face proto from: {self.face_proto_path}")
            
            if not os.path.exists(self.emotion_model_path):
                raise FileNotFoundError(f"Emotion model not found at {self.emotion_model_path}")
            if not os.path.exists(self.face_model_path):
                raise FileNotFoundError(f"Face model not found at {self.face_model_path}")
            if not os.path.exists(self.face_proto_path):
                raise FileNotFoundError(f"Face proto not found at {self.face_proto_path}")
                
            self.emotion_net = cv2.dnn.readNetFromONNX(self.emotion_model_path)
            self.face_net = dnn.readNetFromCaffe(self.face_proto_path, self.face_model_path)
            self.priors = self.define_img_size(self.input_size)
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")

    def define_img_size(self, image_size):
        """Generate prior boxes based on input image size"""
        min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
        strides = [8.0, 16.0, 32.0, 64.0]
        
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in strides]
            feature_map_w_h_list.append(feature_map)

        for _ in range(0, len(image_size)):
            shrinkage_list.append(strides)
            
        priors = self._generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
        return priors

    def _generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        """Generate prior boxes for face detection"""
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h
                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([x_center, y_center, w, h])
        return np.clip(priors, 0.0, 1.0)

    def process_frame(self, frame):
        """Process a single frame for face detection and emotion recognition"""
        img_ori = frame
        width, height = self.input_size
        
        # Prepare input for face detection
        rect = cv2.resize(img_ori, (width, height))
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        self.face_net.setInput(dnn.blobFromImage(rect, 1 / self.image_std, (width, height), 127))
        
        # Detect faces
        boxes, scores = self.face_net.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        
        # Convert detections to boxes
        boxes = self._convert_locations_to_boxes(boxes, self.priors)
        boxes = self._center_form_to_corner_form(boxes)
        boxes, labels, probs = self._predict(img_ori.shape[1], img_ori.shape[0], scores, boxes)
        
        # Process each detected face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []
        
        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1
            
            # Extract and process face for emotion recognition
            face_roi = cv2.resize(gray[y1:y1 + h, x1:x1 + w], (64, 64))
            face_roi = face_roi.reshape(1, 1, 64, 64)
            
            self.emotion_net.setInput(face_roi)
            output = self.emotion_net.forward()
            
            emotion = self.emotion_dict[list(output[0]).index(max(output[0]))]
            results.append({
                'bbox': (x1, y1, x2, y2),
                'emotion': emotion
            })
            
        return results

    def _convert_locations_to_boxes(self, locations, priors):
        """Convert locations to boxes"""
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * self.center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * self.size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)

    def _center_form_to_corner_form(self, locations):
        """Convert center form to corner form"""
        return np.concatenate([
            locations[..., :2] - locations[..., 2:] / 2,
            locations[..., :2] + locations[..., 2:] / 2
        ], len(locations.shape) - 1)

    def _predict(self, width, height, confidences, boxes):
        """Predict face locations"""
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > self.threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
                
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self._hard_nms(box_probs)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
            
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
            
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        
        return (picked_box_probs[:, :4].astype(np.int32), 
                np.array(picked_labels), 
                picked_box_probs[:, 4])

    def _hard_nms(self, box_scores, top_k=-1, candidate_size=200):
        """Perform Non-Maximum Suppression"""
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
                
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self._iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
            indexes = indexes[iou <= self.iou_threshold]
            
        return box_scores[picked, :]

    def _iou_of(self, boxes0, boxes1, eps=1e-5):
        """Calculate Intersection over Union"""
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
        overlap_area = self._area_of(overlap_left_top, overlap_right_bottom)
        area0 = self._area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self._area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def _area_of(self, left_top, right_bottom):
        """Calculate area of boxes"""
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

class EmotionRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Emotion Recognition")
        
        # Initialize detector
        try:
            self.detector = EmotionDetector()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize models: {str(e)}")
            self.window.destroy()
            return
            
        # Create GUI elements
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.window.destroy()
            return
        
        # Define positive and negative emotions
        self.positive_emotions = {'happiness', 'neutral'}
        # All other emotions (sadness, anger, disgust, fear) will be considered negative
            
        # Start processing
        self.process_video()
        
    def process_video(self):
        """Process video frames"""
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB first
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.detector.process_frame(frame)
            
            # Draw results
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                emotion = result['emotion']
                
                # Choose color based on emotion category
                # Using RGB format now (not BGR)
                if emotion in self.positive_emotions:
                    color = (0, 255, 0)  # Green in RGB
                else:
                    color = (0, 0, 255)  # Red in RGB
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Add a filled rectangle behind text for better visibility
                text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 8), (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame, emotion, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
            
            # Frame is already in RGB format for tkinter
            photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
            
            # Update canvas
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.photo = photo
        
        # Schedule next update
        self.window.after(10, self.process_video)
        
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EmotionRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}")
        sys.exit(1)