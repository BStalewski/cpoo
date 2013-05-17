#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import ImageQt

from PyQt4 import QtGui

import algorithms
from thresholding_dialog_ui import Ui_ThresholdingDialog
from ml_em_dialog_ui import Ui_MLEMDialog

class Gui(QtGui.QMainWindow):

    def __init__(self):
        super(Gui, self).__init__()

        self.init_menu_bar()

        self.mainWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QHBoxLayout(self.mainWidget)

        self.src_image_label = QtGui.QLabel()

        self.statusBar()

        self.mainLayout.addWidget(self.src_image_label)
        self.mainLayout.setContentsMargins(5, 5, 5, 5)

        self.setGeometry(300, 300, 250, 250)
        self.setWindowTitle(u'CPOO - segmentacja obrazów')
        self.show()

    def init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu(u'&Plik')
        run_menu = menubar.addMenu(u'&Wykonaj')

        open_image_action = self.create_action(u'Otwórz obraz', u'Ctrl+O',
                                               u'Wczytaj nowy obraz', self.open_image)
        exit_action = self.create_action(u'Exit', u'Ctrl+Q', u'Zakończ program', self.close)
        file_menu.addActions([open_image_action, exit_action])

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
        fname = QtGui.QFileDialog.getOpenFileName(self, u'Wybierz obraz do segmentacji',
                                                  u'', u'Obrazy (*.png *.jpg)')
        if fname != u'':
            pixmap = QtGui.QPixmap(fname)
            self.src_image_label.setPixmap(pixmap)
            self.actions[u'thresholding'].setEnabled(True)
            self.actions[u'ml_em'].setEnabled(True)
            self.src_image_fname = fname

    def thresholding(self):
        dialog = ThresholdingDialog()
        if dialog.exec_():
            dialog_args = dialog.get_values()
            self.execute_algorithm(algorithms.thresholding, dialog_args)

    def ml_em(self):
        dialog = MLEMDialog()
        if dialog.exec_():
            dialog_args = dialog.get_values()
            self.execute_algorithm(algorithms.ml_em, dialog_args)

    def execute_algorithm(self, alg_fun, args):
        fname = unicode(self.src_image_fname)

        new_image = alg_fun(fname, *args)
        self.actions[u'repeat'].setEnabled(True)

        self.prev_algorithm = alg_fun
        self.prev_args = args

    def repeat(self):
        self.execute_algorithm(self.prev_algorithm, self.prev_args)


class ThresholdingDialog(QtGui.QDialog, Ui_ThresholdingDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

    def get_values(self):
        return (self.thresholdsSpinBox.value(), )


class MLEMDialog(QtGui.QDialog, Ui_MLEMDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)

    def get_values(self):
        return (unicode(self.arg1LineEdit.text()), self.arg2SpinBox.value(), )


def main():
    app = QtGui.QApplication(sys.argv)
    gui = Gui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
