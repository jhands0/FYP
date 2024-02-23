import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit

class MyWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        
        self.title = "Heart Disease Classifier"
        self.left = 50
        self.top = 50
        self.width = 400
        self.height = 140

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.textbox = QLineEdit(self)
        self.textbox.move(70, 70)
        self.textbox.resize(280,40)

        self.button = QPushButton("Submit")
        self.button.alignment = Qt.AlignCenter

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.submit)

    def submit(self):
        print("Clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = MyWidget()
    widget.show()

    sys.exit(app.exec())
