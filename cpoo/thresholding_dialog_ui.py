# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'thresholding_dialog.ui'
#
# Created: Sat May 11 12:02:53 2013
#      by: PyQt4 UI code generator 4.10.2-snapshot-74ade0e1faf2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_ThresholdingDialog(object):
    def setupUi(self, ThresholdingDialog):
        ThresholdingDialog.setObjectName(_fromUtf8("ThresholdingDialog"))
        ThresholdingDialog.resize(240, 320)
        self.buttonBox = QtGui.QDialogButtonBox(ThresholdingDialog)
        self.buttonBox.setGeometry(QtCore.QRect(10, 270, 221, 41))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.formLayoutWidget = QtGui.QWidget(ThresholdingDialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 10, 167, 80))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.thresholdsCountLabel = QtGui.QLabel(self.formLayoutWidget)
        self.thresholdsCountLabel.setObjectName(_fromUtf8("thresholdsCountLabel"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.thresholdsCountLabel)
        self.thresholdsSpinBox = QtGui.QSpinBox(self.formLayoutWidget)
        self.thresholdsSpinBox.setMinimum(1)
        self.thresholdsSpinBox.setMaximum(20)
        self.thresholdsSpinBox.setProperty("value", 1)
        self.thresholdsSpinBox.setObjectName(_fromUtf8("thresholdsSpinBox"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.thresholdsSpinBox)

        self.retranslateUi(ThresholdingDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), ThresholdingDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), ThresholdingDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ThresholdingDialog)

    def retranslateUi(self, ThresholdingDialog):
        ThresholdingDialog.setWindowTitle(_translate("ThresholdingDialog", "Dialog", None))
        self.thresholdsCountLabel.setText(_translate("ThresholdingDialog", "Liczba progów", None))

