#!/bin/bash

DEST_DIR="../cpoo"
pyuic4 thresholding_dialog.ui -o "${DEST_DIR}/thresholding_dialog_ui.py"
pyuic4 ml_em_dialog.ui -o "${DEST_DIR}/ml_em_dialog_ui.py"
