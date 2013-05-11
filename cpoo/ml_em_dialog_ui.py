# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ml_em_dialog.ui'
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

class Ui_MLEMDialog(object):
    def setupUi(self, MLEMDialog):
        MLEMDialog.setObjectName(_fromUtf8("MLEMDialog"))
        MLEMDialog.resize(240, 320)
        self.buttonBox = QtGui.QDialogButtonBox(MLEMDialog)
        self.buttonBox.setGeometry(QtCore.QRect(10, 270, 221, 41))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.formLayoutWidget = QtGui.QWidget(MLEMDialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 30, 160, 80))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setMargin(0)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.arg1Label = QtGui.QLabel(self.formLayoutWidget)
        self.arg1Label.setObjectName(_fromUtf8("arg1Label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.arg1Label)
        self.arg1LineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.arg1LineEdit.setObjectName(_fromUtf8("arg1LineEdit"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.arg1LineEdit)
        self.arg2Label = QtGui.QLabel(self.formLayoutWidget)
        self.arg2Label.setObjectName(_fromUtf8("arg2Label"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.arg2Label)
        self.arg2SpinBox = QtGui.QSpinBox(self.formLayoutWidget)
        self.arg2SpinBox.setMinimum(1)
        self.arg2SpinBox.setMaximum(20)
        self.arg2SpinBox.setProperty("value", 1)
        self.arg2SpinBox.setObjectName(_fromUtf8("arg2SpinBox"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.arg2SpinBox)

        self.retranslateUi(MLEMDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), MLEMDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), MLEMDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(MLEMDialog)

    def retranslateUi(self, MLEMDialog):
        MLEMDialog.setWindowTitle(_translate("MLEMDialog", "Dialog", None))
        self.arg1Label.setText(_translate("MLEMDialog", "arg1", None))
        self.arg2Label.setText(_translate("MLEMDialog", "arg2", None))

