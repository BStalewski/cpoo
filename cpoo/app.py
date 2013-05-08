#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gi.repository import Gtk, GdkPixbuf

UI_INFO = u'''
<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='FileOpen' />
      <menuitem action='FileSave' />
      <separator />
      <menuitem action='FileQuit' />
    </menu>
    <menu action='RunMenu'>
      <menuitem action='Thresholding' />
      <menuitem action='ML-EM' />
      <menuitem action='Repeat' />
    </menu>
  </menubar>
</ui>
'''

class GUI(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title=u'CPOO - segmentacja obrazów')

        self.set_default_size(200, 200)

        action_group = Gtk.ActionGroup(u'actions')

        self.add_file_menu_actions(action_group)
        self.add_run_menu_actions(action_group)

        uimanager = self.create_ui_manager()
        uimanager.insert_action_group(action_group)

        menubar = uimanager.get_widget(u'/MenuBar')

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.pack_start(menubar, False, False, 0)

        self.source_image = Gtk.Image()
        self.dest_image = Gtk.Image()
        self.image_box = Gtk.Box()
        self.image_box.pack_start(self.source_image, False, False, 20)
        self.image_box.pack_start(self.dest_image, False, False, 20)

        main_box.pack_start(self.image_box, False, False, 10)

        self.add(main_box)

    def add_file_menu_actions(self, action_group):
        action_group.add_actions([
            (u'FileMenu', None, u'_Plik'),
            (u'FileOpen', Gtk.STOCK_OPEN, u'Otwórz obraz', u'<control>O', None,
             self.on_menu_file_open),
            (u'FileSave', Gtk.STOCK_SAVE_AS, u'Zapisz obraz po segmentacji', u'<control>S', None,
             self.on_menu_file_save),
            (u'FileQuit', Gtk.STOCK_QUIT, u'Zakończ', u'<control>Q', None,
             self.on_menu_file_quit),
        ])

    def add_run_menu_actions(self, action_group):
        action_group.add_actions([
            (u'RunMenu', None, u'_Wykonaj'),
            (u'Thresholding', None, u'Progowanie', u'<control>P', None,
             self.on_menu_run_thresholding),
            (u'ML-EM', None, u'Algorytm ML-EM', u'<control>M', None,
             self.on_menu_run_ml_em),
            (u'Repeat', None, u'Powtórz ostatni', u'<control>R', None,
             self.on_menu_run_repeat),
        ])

    def create_ui_manager(self):
        uimanager = Gtk.UIManager()

        # Throws exception if something went wrong
        uimanager.add_ui_from_string(UI_INFO)

        # Add the accelerator group to the toplevel window
        accelgroup = uimanager.get_accel_group()
        self.add_accel_group(accelgroup)
        return uimanager

    def on_menu_file_open(self, widget):
        dialog = Gtk.FileChooserDialog(u'Wybierz obraz', self, Gtk.FileChooserAction.OPEN);
        dialog.add_button(Gtk.STOCK_CANCEL, 0)
        dialog.add_button(Gtk.STOCK_OK, 1)
        dialog.set_default_response(1)

        filefilter = Gtk.FileFilter()
        filefilter.add_pixbuf_formats()
        dialog.set_filter(filefilter)

        if dialog.run() == 1:
            self.source_image.set_from_file(dialog.get_filename())

        dialog.destroy()

    def on_menu_file_save(self, widget):
        print u'On menu file save'

    def on_menu_file_quit(self, widget):
        Gtk.main_quit()

    def on_menu_run_thresholding(self, widget):
        import array
        colorspace = GdkPixbuf.Colorspace.RGB
        has_alpha = False
        if self.source_image.get_pixbuf():
            bits_per_sample = self.source_image.get_pixbuf().get_bits_per_sample()
            width = self.source_image.get_pixbuf().get_width()
            height = self.source_image.get_pixbuf().get_height()
        else:
            bits_per_sample = 24
            width = 0
            height = 0

        arr = array.array('B')
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(arr, colorspace, False, bits_per_sample, width, height, width * 4)
        print u'on_menu_run_thresholding'

    def on_menu_run_ml_em(self, widget):
        print u'on_menu_run_ml_em'

    def on_menu_run_repeat(self, widget):
        print u'on_menu_run_repeat'


window = GUI()        
window.connect(u'delete-event', Gtk.main_quit)
window.show_all()
Gtk.main()

