import os
import subprocess
import sys

if sys.platform == 'linux2':
    def show_image(file_name):
        subprocess.call(['xdg-open', file_name])
else:
    def show_image(file_name):
        os.startfile(file_name)
