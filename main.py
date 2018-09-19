import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtSql import QSqlQuery
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QMainWindow,\
                            QPushButton, QMessageBox, QAction, QTableWidgetItem, QFileDialog
from PyQt5.uic import loadUi
import sqlite3
import shutil


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi('./views/MainView.ui', self)
        self.setWindowTitle('Employees Finder')
        self.setWindowIcon(QIcon('./imgs/icon.png'))
        self.init_details()
        self.load_employees_data()
        self.load_products_data()
        self.init_handlers()

    def init_details(self):
        avatar = QPixmap('./imgs/avatar.png')
        self.avatar1.setPixmap(avatar)
        self.avatar2.setPixmap(avatar)
        self.avatar3.setPixmap(avatar)
        self.avatar4.setPixmap(avatar)
        self.avatar5.setPixmap(avatar)

    def init_handlers(self):
        self.add_employee.clicked.connect(self.add_employee_handler)
        self.add_product.clicked.connect(self.add_product_handler)
        self.add_employee_image.clicked.connect(self.add_employee_image_handler)
        self.add_product_image.clicked.connect(self.add_product_image_handler)
        self.preview_employee_changes.clicked.connect(self.employee_preview_changes_handler)
        self.preview_product_changes.clicked.connect(self.product_preview_changes_handler)
        self.clear_employee_fields.clicked.connect(self.employee_clear_fields_handler)
        self.clear_product_fields.clicked.connect(self.product_clear_fields_handler)
        self.database_search.clicked.connect(self.database_search_handler)
        self.employees_tableWidget.itemSelectionChanged.connect(self.get_current_employee_row_id)
        self.products_tableWidget.itemSelectionChanged.connect(self.get_current_product_row_id)
        self.tableWidget_database_search.itemSelectionChanged.connect(self.get_current_search_result_row_id)
        self.pushButton_5.clicked.connect(self.train_employee_face_identification)
        self.pushButton_7.clicked.connect(self.train_product_face_identification)
        self.pushButton_2.clicked.connect(self.employee_test_training)
        self.pushButton.clicked.connect(self.product_test_training)

    def employee_test_training(self):
        import cv2
        import numpy as np
        import os 
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarCascades/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #iniciate id counter
        myID = 0
        
        #names related to ids: example ==> Marcelo: id=1,  etc
        names = ['None', 'ahmed khalifa', 'amin', 'mariam', 'W'] 
        
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        
        while True:
        
            ret, img =cam.read()
            #img = cv2.flip(img, -1) # Flip vertically
        
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
        
            for(x,y,w,h) in faces:
        
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
                myID, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                print("print myID :" + str(myID))
                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 65):
                   
                    myID = names[myID]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    myID = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                
                cv2.putText(img, str(myID), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            
            cv2.imshow('camera',img) 
        
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
        
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

    def product_test_training(self):
        pass

    def load_employees_data(self):
        self.employees_tableWidget.setRowCount(0)
        connection = sqlite3.connect('./dbs/mydb.db')
        query = 'SELECT * FROM Employees'
        result = connection.execute(query)
        for row_number, row_data in enumerate(result):
            self.employees_tableWidget.insertRow(row_number)
            for column_number, column_data in enumerate(row_data):
                self.employees_tableWidget.setItem(row_number, column_number, QTableWidgetItem(str(column_data)))
        connection.close()

    def load_products_data(self):
        self.products_tableWidget.setRowCount(0)
        connection = sqlite3.connect('./dbs/mydb.db')
        query = 'SELECT * FROM Products'
        result = connection.execute(query)
        for row_number, row_data in enumerate(result):
            self.products_tableWidget.insertRow(row_number)
            for column_number, column_data in enumerate(row_data):
                self.products_tableWidget.setItem(row_number, column_number, QTableWidgetItem(str(column_data)))
        connection.close()

    def add_employee_handler(self):
        connection = sqlite3.connect('./dbs/mydb.db')
        cur = connection.cursor()
        params = (f"{self.employee_lineEdit_id.text()}", f"{self.employee_lineEdit_name.text()}", f"{self.employee_lineEdit_age.text()}", f"{self.employee_lineEdit_address.text()}", f"{self.employee_lineEdit_role.text()}")
        cur.execute('''INSERT INTO Employees (ID, Name, Age, Address, Role) VALUES (?, ?, ?, ?, ?)''', params)
        connection.commit()
        connection.close()
        self.load_employees_data()

    def add_product_handler(self):
        connection = sqlite3.connect('./dbs/mydb.db')
        cur = connection.cursor()
        params = (f"{self.product_lineEdit_id.text()}", f"{self.product_lineEdit_name.text()}", f"{self.product_lineEdit_price.text()}", f"{self.product_lineEdit_quantity.text()}", f"{self.product_lineEdit_maker.text()}")
        cur.execute('''INSERT INTO Products (ID, Name, Price, Quantity, Maker) VALUES (?, ?, ?, ?, ?)''', params)
        connection.commit()
        connection.close()
        self.load_products_data()

    def add_employee_image_handler(self):
        file_path, file_type = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png)")
        print(file_path)
        shutil.copy2(file_path, f'./imgs/employees/{self.employee_lineEdit_id.text()}.png')
        self.employee_lineEdit_image.setText(f'./imgs/employees/{self.employee_lineEdit_id.text()}.png')

    def add_product_image_handler(self):
        file_path, file_type = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png)")
        print(file_path)
        shutil.copy2(file_path, f'./imgs/products/{self.product_lineEdit_id.text()}.png')
        self.product_lineEdit_image.setText(f'./imgs/products/{self.product_lineEdit_id.text()}.png')

    def employee_clear_fields_handler(self):
        self.tab_2_lineEdit_id.setText("")
        self.tab_2_lineEdit_name.setText("")
        self.tab_2_lineEdit_price.setText("")
        self.tab_2_lineEdit_quantity.setText("")
        self.tab_2_lineEdit_maker.setText("")
        self.tab_2_lineEdit_status.setText("")
        self.avatar2.setPixmap(QPixmap(f'./imgs/avatar.png'))

    def product_clear_fields_handler(self):
        self.tab_4_lineEdit_id.setText("")
        self.tab_4_lineEdit_name.setText("")
        self.tab_4_lineEdit_price.setText("")
        self.tab_4_lineEdit_quantity.setText("")
        self.tab_4_lineEdit_maker.setText("")
        self.tab_4_lineEdit_status.setText("")
        self.avatar4.setPixmap(QPixmap(f'./imgs/avatar.png'))

    def employee_preview_changes_handler(self):
        self.tab_2_lineEdit_id.setText(self.employee_lineEdit_id.text())
        self.tab_2_lineEdit_name.setText(self.employee_lineEdit_name.text())
        self.tab_2_lineEdit_age.setText(self.employee_lineEdit_age.text())
        self.tab_2_lineEdit_address.setText(self.employee_lineEdit_address.text())
        self.tab_2_lineEdit_role.setText(self.employee_lineEdit_role.text())
        self.tab_2_lineEdit_status.setText(self.employee_lineEdit_status.text())
        self.avatar2.setPixmap(QPixmap(f'./imgs/employees/{self.employee_lineEdit_id.text()}.png'))

    def product_preview_changes_handler(self):
        self.tab_4_lineEdit_id.setText(self.product_lineEdit_id.text())
        self.tab_4_lineEdit_name.setText(self.product_lineEdit_name.text())
        self.tab_4_lineEdit_price.setText(self.product_lineEdit_price.text())
        self.tab_4_lineEdit_quantity.setText(self.product_lineEdit_quantity.text())
        self.tab_4_lineEdit_maker.setText(self.product_lineEdit_maker.text())
        self.tab_4_lineEdit_status.setText(self.product_lineEdit_status.text())
        self.avatar4.setPixmap(QPixmap(f'./imgs/products/{self.product_lineEdit_id.text()}.png'))

    def database_search_handler(self):
        import cv2
        import numpy as np
        import os 
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarCascades/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #iniciate id counter
        myID = 0
        
        #names related to ids: example ==> Marcelo: id=1,  etc
        names = ['None', 'ahmed khalifa', 'amin', 'mariam', 'W'] 
        
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        
        while True:
    
            ret, img =cam.read()
            #img = cv2.flip(img, -1) # Flip vertically
        
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
        
            for(x,y,w,h) in faces:
        
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
                myID, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                print("print myID :" + str(myID))
                # Check if confidence is less them 100 ==> "0" is perfect match 
                if(self.database_search_lineEdit.text() == 'back_end'):
                    if(myID == 1 or myID == 2 ):
                        myID = "back_end"
                        confidence = "  {0}%".format(round(100 - confidence))
                        cv2.putText(img, str(myID), (x+5,y-5), font, 1, (255,255,255), 2)
                        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
                elif(self.database_search_lineEdit.text() == 'network_engineering'):
                      if(myID == 3 or myID == 4 ):
                         myID = "network engineering"
                         confidence = "  {0}%".format(round(100 - confidence))
                         cv2.putText(img, str(myID), (x+5,y-5), font, 1, (255,255,255), 2)
                         cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)      
                
            cv2.imshow('camera',img) 
        
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
        
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
            
        

    def get_current_employee_row_id(self):
        items = self.employees_tableWidget.selectedItems()
        # print(str(items[0].text()))
        connection = sqlite3.connect('./dbs/mydb.db')
        cursor = connection.cursor()
        query_employees = f"SELECT * FROM Employees WHERE ID='{str(items[0].text())}'"
        cursor.execute(query_employees)
        for row in cursor:
            # print(row)
            eid, name, age, address, role = row
            self.tab_1_lineEdit_id.setText(str(eid))
            self.tab_1_lineEdit_name.setText(str(name))
            self.tab_1_lineEdit_age.setText(str(age))
            self.tab_1_lineEdit_address.setText(str(address))
            self.tab_1_lineEdit_role.setText(str(role))
            self.avatar1.setPixmap(QPixmap(f'./imgs/employees/{eid}.png'))
        connection.close()

    def get_current_product_row_id(self):
        items = self.products_tableWidget.selectedItems()
        # print(str(items[0].text()))
        connection = sqlite3.connect('./dbs/mydb.db')
        cursor = connection.cursor()
        query_products = f"SELECT * FROM Products WHERE ID='{str(items[0].text())}'"
        cursor.execute(query_products)
        for row in cursor:
            # print(row)
            pid, name, price, quantity, maker = row
            self.tab_3_lineEdit_id.setText(str(pid))
            self.tab_3_lineEdit_name.setText(str(name))
            self.tab_3_lineEdit_price.setText(str(price))
            self.tab_3_lineEdit_quantity.setText(str(quantity))
            self.tab_3_lineEdit_maker.setText(str(maker))
            self.avatar3.setPixmap(QPixmap(f'./imgs/products/{pid}.png'))
        connection.close()

    def get_current_search_result_row_id(self):
        items = self.tableWidget_database_search.selectedItems()
        print(str(items[0].text()))
        connection = sqlite3.connect('./dbs/mydb.db')
        cursor = connection.cursor()
        query_employees = f"SELECT * FROM Employees WHERE ID='{str(items[0].text())}'"
        query_products = f"SELECT * FROM Products WHERE ID='{str(items[0].text())}'"
        cursor.execute(query_employees)

        for row in cursor:
            print(row)
            eid, name, age, address, role = row
            self.tab_5_lineEdit_id.setText(str(eid))
            self.tab_5_lineEdit_name.setText(str(name))
            self.avatar5.setPixmap(QPixmap(f'./imgs/employees/{eid}.png'))

        # cursor = connection.cursor()
        # cursor.execute(query_products)
        # for row in cursor:
        #     # print(row)
        #     pid, name = row
        #     self.tab_5_lineEdit_id.setText(str(pid))
        #     self.tab_5_lineEdit_name.setText(str(name))
        #     self.avatar5.setPixmap(QPixmap(f'./imgs/products/{pid}.png'))

        connection.close()
    
    def train_employee_face_identification(self):
        import cv2
        import numpy as np
        import os
        from PIL import Image
        import os
        
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

        face_detector = cv2.CascadeClassifier('haarCascades\haarcascade_frontalface_default.xml')

        # For each person, enter one numeric face id
        face_id = self.employee_lineEdit_id.text()

        #print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0
        
        while(True):

            ret, img = cam.read()
            #img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                
                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 1000: # Take 30 face sample and stop video
                break
        
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        
        # Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarCascades\haarcascade_frontalface_default.xml");

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                #print(id)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
                    #print(id)
            return faceSamples,ids

        # print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        self.employee_lineEdit_status.setText('[INFO] Training faces. It will take a few seconds. Wait ...')
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        # print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        self.employee_lineEdit_status.setText(f'[INFO] {np.unique(ids)} faces trained.')
        

    
    def train_product_face_identification(self):
        pass
    


app = QApplication(sys.argv)
widget = MyWindow()
widget.show()
sys.exit(app.exec())





