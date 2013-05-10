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

        open_image_action = self.create_action(u'Otwórz obraz', u'Ctrl+O',
                                               u'Wczytaj nowy obraz', self.open_image)
        save_image_action = self.create_action(u'Zapisz obraz po segmentacji', u'Ctrl+S',
                                               u'Zapisz przetworzony obraz', self.save_image, False)
        exit_action = self.create_action(u'Exit', u'Ctrl+Q', u'Zakończ program', self.close)
        file_menu.addActions([open_image_action, save_image_action, exit_action])

        thresholding_action = self.create_action(u'Progowanie', u'Ctrl+P', u'Wykonaj algorytm progowania',
                                                 self.thresholding, False)
        ml_em_action = self.create_action(u'Algorytm ML-EM', u'Ctrl+M',
                                          u'Wykonaj algorytm ML-EM', self.ml_em, False)
        repeat_action = self.create_action(u'Powtórz ostatni algorytm', u'Ctrl+R',
                                           u'Wykonaj ponownie poprzedni algorytm z tym samymi ustawieniami',
                                           self.repeat, False)
        run_menu.addActions([thresholding_action, ml_em_action, repeat_action])

        self.actions = {
            u'open': open_image_action,
            u'save': save_image_action,
            u'exit': exit_action,
            u'thresholding': thresholding_action,
            u'ml_em': ml_em_action,
            u'repeat': repeat_action,
        }

    def create_action(self, name, shortcut, status_tip, callback, enabled=True):
        action = QtGui.QAction(name, self)
        action.setShortcut(shortcut)
        action.setStatusTip(status_tip)
        action.triggered.connect(callback)
        action.setEnabled(enabled)
        return action

    def open_image(self):
        self.src_fname = QtGui.QFileDialog.getOpenFileName(self, u'Wybierz obraz do segmentacji',
                                                           u'', u'Obrazy (*.png *.jpg)')
        pixmap = QtGui.QPixmap(self.src_fname)
        self.src_label.setPixmap(pixmap)
        self.actions[u'thresholding'].setEnabled(True)
        self.actions[u'ml_em'].setEnabled(True)

    def save_image(self):
        self.segmented_image_fname = QtGui.QFileDialog.getSaveFileName(self, u'Zapisz obraz po segmentacji',
                                                                       u'', u'Obrazy (*.png *.jpg)')
        self.segmented_image.save(self.segmented_image_fname)

    def after_algorithm(self, segmented_image):
        self.segmented_image = ImageQt.ImageQt(segmented_image)
        # if something bad happens here, try to uncomment
        #self.segmented_image = QtGui.QImage(self.segmented_image)
        segmented_image_pixmap = QtGui.QPixmap.fromImage(self.segmented_image)
        self.segmented_image_label.setPixmap(segmented_image_pixmap)
        self.actions[u'save'].setEnabled(True)
        self.actions[u'repeat'].setEnabled(True)

    def thresholding(self):
        new_image = algorithms.thresholding(unicode(self.src_fname))
        self.last_method = self.thresholding
        self.after_algorithm(new_image)

    def ml_em(self):
        new_image = algorithms.ml_em(unicode(self.src_fname))
        self.last_method = self.ml_em
        self.after_algorithm(new_image)

    def repeat(self):
        self.last_method()


def main():
    app = QtGui.QApplication(sys.argv)
    gui = Gui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
