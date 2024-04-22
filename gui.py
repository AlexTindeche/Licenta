import sys

from numpy import block
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QTabWidget, QFileDialog, QToolButton, QFileDialog, QLineEdit, QHBoxLayout
from PySide6.QtGui import QFont, QTextBlockFormat, QTextCursor, QImage, QTextDocument, QTextImageFormat
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Management System")
        self.setMinimumSize(QSize(1000, 700))
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)

        self.setStyleSheet("""
            QMainWindow {
                border: 2px solid gray;
                background-color: gray;
                box-shadow: 10px 10px 5px grey;
                           
            }
        """)

        self.notebook = QTabWidget()
        self.tab = QWidget()
        self.notebook.addTab(self.tab, "Home")

        self.textEdit = QTextEdit(self.tab)
        self.textEdit.setReadOnly(True)
        self.textEdit.setFont(QFont('Arial', 14))
        self.tab1Layout = QVBoxLayout(self.tab)
        self.tab1Layout.addWidget(self.textEdit)


        self.display_paragraph("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed mi elit, cursus vitae pretium quis, pulvinar in tortor. Vestibulum ante lorem, luctus eget tortor sed, elementum faucibus orci. Quisque at consequat ex, a mattis elit. Mauris vel purus at ligula ornare semper. Nullam iaculis arcu quis congue vulputate. Cras in efficitur massa. Aliquam ut semper augue, sed egestas tortor. In venenatis venenatis mauris, ac cursus ligula porta nec. Duis semper tempus sodales. Cras at neque ac augue ultrices suscipit. Mauris sed tincidunt est. In egestas id diam sit amet ultrices. Praesent vel arcu ultricies, convallis sapien pellentesque, scelerisque ipsum. ", self.textEdit)
        self.display_paragraph("In lobortis sollicitudin cursus. Sed tincidunt elit eget dui sodales, at congue diam ultricies. Duis urna turpis, tincidunt vulputate finibus id, consectetur non nibh. Morbi ut elit cursus nibh elementum cursus sed quis libero. Donec interdum molestie velit sit amet pulvinar. Etiam massa ex, imperdiet quis urna id, iaculis ultrices eros. Vivamus lacinia, mauris vulputate lobortis interdum, turpis nibh consequat nisi, ut finibus nisl eros in nisi. Aenean convallis libero ut venenatis feugiat. Morbi in mi semper, fringilla metus sed, tincidunt mauris. In sed leo sit amet mi vehicula sodales. Aliquam sodales erat nibh, vitae sollicitudin magna suscipit gravida. Phasellus non fermentum massa. Integer in auctor tortor. Integer ut dolor eu dui faucibus efficitur eu et sem. Praesent rhoncus finibus pharetra. ", self.textEdit)
        self.display_paragraph("Curabitur dictum, sapien ac sagittis suscipit, ex sem sagittis sapien, vitae eleifend arcu augue sed massa. Aliquam aliquam eleifend scelerisque. Maecenas nec congue eros, a condimentum lorem. Maecenas dapibus vel nulla eu hendrerit. Quisque condimentum massa risus, quis tempus ex fermentum ut. Aliquam vel finibus metus. Proin mollis, sem vel porta tincidunt, mi eros cursus libero, in iaculis metus urna quis mi. Cras eu rhoncus quam. In posuere tempus molestie. Nunc pretium nibh lorem, a blandit massa vehicula vitae. Ut at ligula id odio tristique condimentum. Sed non neque consectetur, fringilla elit non, ultrices elit. Vestibulum nisi neque, suscipit sed sollicitudin sed, dictum in lacus. ", self.textEdit)
        self.display_paragraph("Donec elementum lorem in porta finibus. Sed lacinia egestas nisl ac vehicula. Phasellus consequat sapien tempor risus rhoncus blandit. In hac habitasse platea dictumst. Nulla facilisi. Nunc vitae nisi eget ipsum aliquam vulputate. Sed ut lorem sed felis fermentum finibus in a lectus. ", self.textEdit)
        self.display_paragraph("Proin pretium, magna sed feugiat pulvinar, nisi urna lacinia ligula, ut lacinia nisi lacus vitae libero. Etiam efficitur aliquam nisi, at rutrum massa blandit id. Pellentesque a magna auctor, sagittis tortor eget, suscipit lacus. Duis ac viverra justo. Interdum et malesuada fames ac ante ipsum primis in faucibus. Quisque vestibulum, nibh et porttitor imperdiet, nulla sapien mollis mauris, et ultricies dolor nibh id sem. Vivamus eleifend elit eget neque facilisis, vitae ullamcorper mi fringilla. Integer porttitor tortor nec magna pulvinar sodales. Phasellus lacus nibh, scelerisque iaculis mauris non, interdum finibus sem. Duis maximus risus nisl, eu semper massa egestas ac. Cras suscipit purus sit amet sodales auctor. ", self.textEdit)

        # Tab 2
        self.tab2 = QWidget()
        self.notebook.addTab(self.tab2, "Video analyzer")
        layout.addWidget(self.notebook)

        # Add text to tab 2
        self.textEdit2 = QTextEdit(self.tab2)
        self.textEdit2.setReadOnly(True)
        self.textEdit2.setFont(QFont('Arial', 14))
        self.tab2Layout = QVBoxLayout(self.tab2)
        self.tab2Layout.addWidget(self.textEdit2)   


        self.display_paragraph("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed mi elit, cursus vitae pretium quis, pulvinar in tortor. Vestibulum ante lorem, luctus eget tortor sed, elementum faucibus orci. Quisque at consequat ex, a mattis elit. Mauris vel purus at ligula ornare semper. Nullam iaculis arcu quis congue vulputate. Cras in efficitur massa. Aliquam ut semper augue, sed egestas tortor. In venenatis venenatis mauris, ac cursus ligula porta nec. Duis semper tempus sodales. Cras at neque ac augue ultrices suscipit. Mauris sed tincidunt est. In egestas id diam sit amet ultrices. Praesent vel arcu ultricies, convallis sapien pellentesque, scelerisque ipsum. ", self.textEdit2)

        # Display image on tab 1
        self.display_image("gui_utils/images/DIGI-Native-content-image.jpg", self.textEdit)

        self.lineLayout = QHBoxLayout()

        # Add label to the layout
        self.outputFileLabel = QLineEdit(self.tab2)
        self.outputFileLabel.setText('Input file path')
        self.outputFileLabel.setReadOnly(True)
        self.outputFileLabel.setFixedWidth(100)
        self.outputFileLabel.setAlignment(Qt.AlignLeft)
        self.lineLayout.addWidget(self.outputFileLabel)

        # Create a QLineEdit
        self.outputFileLineEdit = QLineEdit(self.tab2)
        self.outputFileLineEdit.setPlaceholderText('Input file path')
        self.lineLayout.addWidget(self.outputFileLineEdit)

        # Create a QToolButton
        self.inputFileButton = QToolButton(self.tab2)
        self.inputFileButton.setObjectName('inputFileButton')
        self.inputFileButton.setText('Select video file')
        self.lineLayout.addWidget(self.inputFileButton)

        # Add the QHBoxLayout to the QVBoxLayout
        self.tab2Layout.addLayout(self.lineLayout)



        # Connect the button's clicked signal to the onInputFileButtonClicked slot
        self.inputFileButton.clicked.connect(self.onInputFileButtonClicked)



    def display_paragraph(self, text, textEdit):
        cursor = QTextCursor(textEdit.document())
        cursor.movePosition(QTextCursor.End)
        blockFormat = QTextBlockFormat()
        blockFormat.setBottomMargin(10)
        blockFormat.setLineHeight(3.3, 4)
        blockFormat.setLeftMargin(5)
        cursor.insertBlock(blockFormat)
        cursor.insertText(text)

    def display_image(self, image_path, textEdit):
        image = QImage(image_path)
        # Resize the image
        cursor = QTextCursor(textEdit.document())
        cursor.movePosition(QTextCursor.End)

        # Create a block format for the image and set its alignment to center
        blockFormat = QTextBlockFormat()
        blockFormat.setAlignment(Qt.AlignCenter)  # Center the block
        blockFormat.setBottomMargin(10)
        cursor.insertBlock(blockFormat)

        # Now insert the image
        imageFormat = QTextImageFormat()
        imageFormat.setName(image_path)
        imageFormat.setWidth(700)
        cursor.insertImage(imageFormat)

        # If you want to add more text after the image, ensure to reset block formatting
        blockFormat = QTextBlockFormat()  # Reset to default formatting
        cursor.insertBlock(blockFormat)

    def onInputFileButtonClicked(self):
        # Get input file path
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', 'Video Files (*.mp4 *.avi)')
        self.outputFileLineEdit.setText(filename)
        print(filename)



app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
