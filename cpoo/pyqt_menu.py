#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import Image
import ImageQt

from PyQt4 import QtCore
from PyQt4 import QtGui

import algorithms

class Gui(QtGui.QMainWindow):

    def __init__(self):
        super(Gui, self).__init__()

        self.init_menu_bar()

        self.mainWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QHBoxLayout(self.mainWidget)

        self.src_label = QtGui.QLabel()
        self.segmented_image_label = QtGui.QLabel()

        self.statusBar()

        self.mainLayout.addWidget(self.src_label)
        self.mainLayout.addWidget(self.segmented_image_label)

        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle(u'CPOO - segmentacja obrazów')
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
        self.src_fname = QtGui.QFileDialog.getOpenFileName(self, u'Wybierz obraz do segmentacji',
                                                           u'', u'Obrazy (*.png *.jpg)')
        pixmap = QtGui.QPixmap(self.src_fname)
        self.src_label.setPixmap(pixmap)

    def save_image(self):
        self.segmented_image_fname = QtGui.QFileDialog.getSaveFileName(self, u'Zapisz obraz po segmentacji',
                                                                       u'', u'Obrazy (*.png *.jpg)')
        self.segmented_image.save(self.segmented_image_fname)

    def thresholding(self):
        new_image = algorithms.thresholding(unicode(self.src_fname))
        self.segmented_image = ImageQt.ImageQt(new_image)
        # if something bad happens here, try to uncomment
        #self.segmented_image = QtGui.QImage(self.segmented_image)
        segmented_image_pixmap = QtGui.QPixmap.fromImage(self.segmented_image)
        self.segmented_image_label.setPixmap(segmented_image_pixmap)
        self.last_method = self.thresholding

    def ml_em(self):
        new_image = algorithms.ml_em(unicode(self.src_fname))
        self.segmented_image = ImageQt.ImageQt(new_image)
        # if something bad happens here, try to uncomment
        #self.segmented_image = QtGui.QImage(self.segmented_image)
        segmented_image_pixmap = QtGui.QPixmap.fromImage(self.segmented_image)
        self.segmented_image_label.setPixmap(segmented_image_pixmap)
        self.last_method = self.ml_em

    def repeat(self):
        self.last_method()


def main():
    app = QtGui.QApplication(sys.argv)
    gui = Gui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
