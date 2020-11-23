#!/usr/bin/env python

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import cv2
import snake as s

Window.fullscreen = 'auto'

filePath = None

class MainScreen(Screen):
    pass

class FileChooserScreen(Screen):
    pass

class ScreenManagement(ScreenManager):
    pass

presentation = Builder.load_file("app.kv")

class SnakeApp(App):
    def setPath(self, selectedImagePath):
        global filePath
        filePath = selectedImagePath
        print(filePath)

    def processImageButtonClick(self):
        # Loads the desired image
        global filePath

        if(filePath == None):
            print("Select Image First")
            popup = Popup(title='Error',content=Label(text='Image file not selected'),size_hint=(None, None), size=(400, 400))
            popup.open()
            return

        image = cv2.imread( filePath, cv2.IMREAD_COLOR )

        snake = s.Snake( image, closed = True )

        snake_window_name = "Snakes"
        cv2.namedWindow( snake_window_name )
        snake.set_alpha(self.root.get_screen('mainscreen').ids.alpha.value)
        snake.set_beta(self.root.get_screen('mainscreen').ids.beta.value)
        snake.set_delta(self.root.get_screen('mainscreen').ids.delta.value)
        snake.set_w_line(self.root.get_screen('mainscreen').ids.wline.value)
        snake.set_w_edge(self.root.get_screen('mainscreen').ids.wedge.value)
        snake.set_w_term(self.root.get_screen('mainscreen').ids.wterm.value)

        while( True ):
            snakeImg = snake.visualize()
            cv2.imshow( snake_window_name, snakeImg )
            snake_changed = snake.step()
            k = cv2.waitKey(33)
            if k == 27:
                break

        cv2.destroyAllWindows()

    def build(self):
        return presentation

SnakeApp().run()