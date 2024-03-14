import sys
from PySide6.QtWidgets import ( QPushButton, QApplication,
    QVBoxLayout, QMainWindow,QComboBox, QGroupBox
                               , QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QHBoxLayout , QWidget,QSlider)
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtCore import Slot, QThread, Signal, Qt
from PySide6.QtGui import QImage, QAction, QKeySequence, QPixmap

import time
import os
import cv2



class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True
        self.f_detection = False
        self.sharpness_inverse=0
        self.e_detection = False

    def set_file(self, fname):
        # The data comes with the 'opencv-python' module
        self.trained_file = os.path.join(cv2.data.haarcascades, fname)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            color_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #if face detection is enabled
            if self.f_detection==True:
                color_frame=self.face_detection(frame)
            
            #if edge detection is enabled
            elif self.e_detection==True and self.sharpness_inverse>0:
                color_frame=self.edge_detection(frame)
            
            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(720, 480, Qt.KeepAspectRatio)
            if self.status==True:
                stored_image=scaled_img
                # Emit signal
                self.updateFrame.emit(scaled_img)
            else :
                self.updateFrame.emit(stored_image)
        sys.exit(-1)

    def face_detection(self,frame):
        cascade = cv2.CascadeClassifier(self.trained_file)

        # Reading frame in gray scale to process the pattern
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30))

        # Drawing green rectangle around the pattern
        for (x, y, w, h) in detections:
            pos_ori = (x, y)
            pos_end = (x + w, y + h)
            color = (0, 255, 0)
            cv2.rectangle(frame, pos_ori, pos_end, color, 2)

        # Reading the image in RGB to display it
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return color_frame
    
    def edge_detection(self,frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
        
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=self.sharpness_inverse) # Combined X and Y Sobel Edge Detection
        
        sobelxy= cv2.normalize(sobelxy, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_frame= cv2.cvtColor(sobelxy, cv2.COLOR_GRAY2RGB)
        return color_frame
    
        

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Patterns detection")
        self.setGeometry(0, 0, 800, 500)

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=QApplication.quit)  # noqa: F821
        self.menu_file.addAction(exit)

        

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(720, 480)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)
        
        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self, shortcut=QKeySequence(QKeySequence.HelpContents),
                        triggered=self.th.terminate())  # noqa: F821
        self.menu_about.addAction(about)
        # Model group
        self.group_model = QGroupBox("Trained model")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()

        self.combobox = QComboBox()
        for xml_file in os.listdir(cv2.data.haarcascades):
            if xml_file.endswith(".xml"):
                self.combobox.addItem(xml_file)

        model_layout.addWidget(QLabel("File:"), 10)
        model_layout.addWidget(self.combobox, 90)
        self.group_model.setLayout(model_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button3 = QPushButton("Pause")
        self.button4 = QPushButton("Face Detect")
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button4.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.button4)
        buttons_layout.addWidget(self.button3)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)
        
        right_layout = QHBoxLayout()
        right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 1)
        
        self.slider_group = QGroupBox("Edge Detection")
        self.slider_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        slider_layout = QHBoxLayout()
        self.slider=QSlider()
        self.slider.setMinimum(1)
        self.slider.setMaximum(15)
        self.slider.setTickInterval(2)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        slider_layout.addWidget(self.slider,3)
        slider_layout.addLayout(right_layout,1)
           
        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(slider_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)
        self.button3.clicked.connect(self.pause)
        self.button3.setEnabled(False)
        self.button4.clicked.connect(self.face_detection)
        self.slider.valueChanged.connect(self.edge_detection)
        self.combobox.currentTextChanged.connect(self.set_model)

    @Slot()
    def set_model(self, text):
        self.th.set_file(text)
    
    @Slot()
    def edge_detection(self):
        if self.slider.value()%2==1:
            self.th.sharpness_inverse=self.slider.value()
        if self.th.sharpness_inverse>1:
            self.th.e_detection= True

        else:
            self.th.e_detection=False
    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.th.status = False
        
        self.th.terminate()
        
        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def pause(self):
        print("Pause successul...")
        self.button3.setEnabled(False)
        self.button1.setEnabled(True)
        
        self.th.status = False
        
        
        
    @Slot()
    def face_detection(self):
        if  self.th.f_detection== False:
            self.th.f_detection= True
        else:
            self.th.f_detection=False
    @Slot()
    def start(self):
        print("Starting...")
        self.button3.setEnabled(True)
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th.set_file(self.combobox.currentText())
        self.th.status=True
        self.th.start()
        
        
        
        

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))



def check_camera_availability():
    if len(QMediaDevices.videoInputs())>0:
        return True
    else:
        return False

        
if __name__ == '__main__':
    
    #check camera 
    if check_camera_availability():
        pass
    else:
        sys.exit()
    
    # Create and run the qtapplication
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
    
    