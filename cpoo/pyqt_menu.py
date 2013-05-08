#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui

class Example(QtGui.QMainWindow):
    
    def __init__(self):
        super(Example, self).__init__()
        
        self.initUI()
        
    def initUI(self):               
        self.init_menu_bar()
        
        textEdit = QtGui.QTextEdit()
        self.setCentralWidget(textEdit)

        self.statusBar()

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle(u'Main window')    
        self.show()

    def init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu(u'&Plik')
        run_menu = menubar.addMenu(u'&Wykonaj')

        open_image_action = QtGui.QAction(u'Otwórz obraz', self)
        open_image_action.setShortcut(u'Ctrl+O')
        open_image_action.setStatusTip(u'Wczytaj nowy obraz')
        open_image_action.triggered.connect(self.open_image)

        save_image_action = QtGui.QAction(u'Zapisz obraz po segmentacji', self)
        save_image_action.setShortcut(u'Ctrl+S')
        save_image_action.setStatusTip(u'Zapisz przetworzony obraz')
        save_image_action.triggered.connect(self.save_image)

        exit_action = QtGui.QAction(u'Exit', self)
        exit_action.setShortcut(u'Ctrl+Q')
        exit_action.setStatusTip(u'Zakończ program')
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_image_action)
        file_menu.addAction(save_image_action)
        file_menu.addAction(exit_action)

        thresholding_action = QtGui.QAction(u'Progowanie', self)
        thresholding_action.setShortcut(u'Ctrl+P')
        thresholding_action.setStatusTip(u'Wykonaj algorytm progowania')
        thresholding_action.triggered.connect(self.thresholding)

        ml_em_action = QtGui.QAction(u'Algorytm ML-EM', self)
        ml_em_action.setShortcut(u'Ctrl+M')
        ml_em_action.setStatusTip(u'Wykonaj algorytm ML-EM')
        ml_em_action.triggered.connect(self.ml_em)

        repeat_action = QtGui.QAction(u'Powtórz ostatni algorytm', self)
        repeat_action.setShortcut(u'Ctrl+R')
        repeat_action.setStatusTip(u'Wykonaj ponownie poprzedni algorytm z tym samymi ustawieniami')
        repeat_action.triggered.connect(self.repeat)

        run_menu.addAction(thresholding_action)
        run_menu.addAction(ml_em_action)
        run_menu.addAction(repeat_action)


    def open_image(self):
        print u'open image'
        
    def save_image(self):
        print u'save image'

    def thresholding(self):
        print u'thresholding'
        
    def ml_em(self):
        print u'ML-EM'

    def repeat(self):
        print u'repeat'
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
