
from kivy.config import Config
Config.set('graphics', 'width', '1100')
Config.set('graphics', 'height', '700')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget  
from kivy.uix.label import Label
from kivy.graphics import Color, Line
from PIL import Image, ImageDraw




class DrawingWidget(Widget):
    def __init__(self, **kwargs): #The __init__ method initializes the widget and creates a blank image using the Image class from PIL.
        super(DrawingWidget, self).__init__(**kwargs)
        self.image = Image.new("RGB", (500, 500), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def on_touch_down(self, touch): # It draws a white line on the canvas at the position of the touch event using the Line class
        with self.canvas:
            Color(1, 1, 1)
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=2)
            self.draw.line([touch.x, touch.y, touch.x, touch.y], fill=(0, 0, 0), width=2)

    def on_touch_move(self, touch): #This method is called when a touch event (mouse movement or dragging on the screen) occurs on the widget. It updates the points attribute of the Line object to draw a continuous line as the touch moves.
        touch.ud["line"].points += (touch.x, touch.y)
        self.draw.line([touch.x, touch.y, touch.x, touch.y], fill=(0, 0, 0), width=2)

    def clear_canvas(self):
        self.canvas.clear()
        self.image = Image.new("RGB", (500, 500), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)



from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

model_json = open('model.json', 'r')
loaded_model_json = model_json.read()
model_json.close()
model = model_from_json(loaded_model_json)


model.load_weights("model_weights.h5")


labels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','+','-','x']



from sympy import Eq, sympify, solve


''' The Solver class is defined to convert and solve the equation using the SymPy library. It has methods to convert the equation into general form and solve the equation.'''

class Solver:

    def __init__(self, equation):
        self.equation = str(equation)
        self.leftEqu = []

    def convertEquationIntoGeneralForm(self): # method is responsible for converting the equation into a general form.    
        left_Side, right_Side = '', ''
        equal_to_Indx = self.equation.index('=')
        left_Side = self.equation[0:equal_to_Indx]
        right_Side = self.equation[equal_to_Indx+1:len(self.equation)]
            
        if len(right_Side) > 0 and (right_Side[0].isalpha() or right_Side[0].isdigit()):
                right_Side = '+' + right_Side

        for i in range(0, len(right_Side)):  #In our example, after the loop, right_Side will be "-1" and left_Side will be "x^2 - 5*x + 5-1".
            if right_Side[i] == '+':
                right_Side = right_Side[0:i] + '-' + right_Side[i+1:len(right_Side)]
            elif right_Side[i] == '-':
                right_Side = right_Side[0:i] + '+' + right_Side[i+1:len(right_Side)]
            left_Side += right_Side[i]

        self.equation = left_Side + '=' + '0'  # In our example, self.equation will be "x^2 - 5*x + 5-1=0" and self.leftEqu will be "x^2 - 5*x + 5-1".
        self.leftEqu = left_Side


    def solveEquation(self):

        self.convertEquationIntoGeneralForm()
        sympy_eq = sympify("Eq(" + self.equation.replace("=", ",") + ")") #convert the modified equation (self.equation) into a SymPy expression.
        roots = solve(sympy_eq)  # solve is smpy library function
        print("root",roots)
        return roots




def predictFromModel(arr): #The predictFromModel function takes an array as input and predicts the class label using the trained CNN model. It uses the np.argmax function to obtain the index of the predicted class.
    result = np.argmax(model.predict(arr), axis=-1)
    return result
    

#The diplay function is responsible for updating the GUI labels with the original equation and its roots.
def display(self,equation="Error no equal to found ", roots="Please try again"):
    self.equation_label.text = 'Your Equation is: ' + equation
    self.roots_label.text = 'Result: ' + roots + '\n'    
''' 
Defining the solution function:

The solution function is called when the "Predict" button is clicked. It performs the following steps:
- Reads and preprocesses the saved canvas image using OpenCV functions.
-Finds the contours of the handwritten digits and operations in the image.
-Resizes and reshapes each contour image to match the input size expected by the model.
-Calls the predictFromModel function to predict the class labels for each contour image.
-Converts the predicted labels into a string equation by concatenating the corresponding labels.
-Creates an instance of the Solver class with the string equation and calls the solveEquation method to obtain the roots of the equation.
-Displays the original equation and its roots in the GUI.'''

def find_solution(self):  
    try:
        img = cv2.imread('can.png',cv2.IMREAD_GRAYSCALE)

        # Convert the image to grayscale
        # img= ~image  #Contour detection on the image above will only result in a contour outlining the edges of the image. This is because the **`cv2.findContours()`** function expects the foreground to be white, and the background to be black

        # Apply thresholding to create a binary image (to convert into pure black and white image)
        ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) 

        # Find contours
        ''' In simple terms, a contour is a curve that represents the boundary of an object in an image. It is a set of connected points that form a closed shape.

        When working with image processing and computer vision tasks, contours are commonly used for object detection, shape analysis, and region of interest extraction. By finding contours in an image, you can identify and isolate individual objects or shapes present in the image.'''
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # mode is external mean only external contor is detected
        cnt = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        print("contours",cnt)
        # Display the total number of contours found.
        print("Number of contours found = {}".format(len(cnt)))

        # Read the image in color mode for drawing purposes.
        image1 = cv2.imread('can.png')  


        # Draw all the contours.

        cv2.drawContours(image1, contours, -1, (0,255,0), 3) # here -1 means all contors , 0 means first contor and so on


        # plt.figure(figsize=[10,10])
        # plt.imshow(image1[:,:,::-1]);plt.axis("off"); # this is because cv2 read image in BGR and matplot read in RGB , so we need to change
        # plt.show()

        # Loop over the contours

        # for i in range(len(contours)):
                
        #         # Draw the ith contour
        #         image1 = cv2.imread('can.png') 
        #         print(i)
        #         cv2.drawContours(image1, contours, i, (0,255,0), 3)

        #         # Add a subplot to the figure
        #         plt.subplot(3, 3, i+1)  # 3X3 no. of images maximum

        #         # Turn off the axis
        #         plt.axis("off");plt.title('contour ' +str(i+1))

        #         # Display the image in the subplot
        #         plt.imshow(image1[:,:,::-1])
        # plt.show()




        img_data = []
        rects = []
        for c in cnt :
                x, y, w, h = cv2.boundingRect(c)
                rect = [x, y, w, h]
                rects.append(rect)
        final_rect = [i for i in rects]

        print("r",rects)
        print("fr",final_rect)


        ''' After finding the boundary coordinates of the image, the following code extracts the regions of interest (ROI) from the image based on the bounding rectangles.'''
        i=0
        for r in final_rect:
                x,y,w,h = r[0],r[1],r[2],r[3]
                img = thresh[y:y+h+10, x:x+w+10]
                img = cv2.resize(img, (28, 28))
                # Add a subplot to the figure
                # plt.subplot(3, 4, i+1)  
                # plt.imshow(img,cmap='hot')
                img = np.reshape(img, (1, 28, 28)) 
                img_data.append(img)  
                i=i+1

        # plt.show()
                   
        Equation=[]
        for i in range(len(img_data)):
                    img_data[i] = np.array(img_data[i])
                    img_data[i] = img_data[i].reshape(-1, 28, 28, 1)
                    result=predictFromModel(img_data[i])
                    i=result[0]
                    Equation.append(labels[i])  # initialized as an empty list to store the predicted labels.
        print("main equation:-",Equation)    
        st_Equation=""
        for i in range(len(Equation)):
                a=Equation[i]
                if(a.isdigit()==False and a.isalpha()==False and i<len(Equation) and i>0):
                    # if(a==Equation[i+1]=='-'):
                    #     st_Equation+='='
                    if(a==st_Equation[-1]=='-'):
                        st_Equation=st_Equation[:-1]
                        st_Equation+='='    
                    else:
                        st_Equation+=a
                if(a.isalpha()==True):
                    if(i>0):
                        if(Equation[i-1].isdigit()): # 5X means 5 x X
                            st_Equation+="*"+a
                        else:
                            st_Equation+=a
                    else:
                        st_Equation+=a
                if(a.isdigit()==True):
                    if(i>0):
                        if(Equation[i-1].isdigit()): 
                            st_Equation+=a
                        elif(Equation[i-1].isalpha()):  # X2 means X ^2
                            st_Equation+="^"+a
                        else:
                            st_Equation+=a
                    else:
                        st_Equation+=a
                
        new_String_Equation=""
        l=list(st_Equation)   
        for i in range(len(l)):
                if(l[i]=="="):
                    new_String_Equation=l[:i+1]+l[i+2:]
        print("string_Equation",st_Equation)
        print("newstr",new_String_Equation)
        equation=""
        # for i in new_String_Equation:
        #         eqution+=i
        for i in st_Equation:
                equation+=i        
        sol=Solver(equation)
        print("equ",equation)        
        str1=''        
        roots=sol.solveEquation()
        st=[]
        for i in roots:
            i=str(i)
            st.append(i)
        
        str1=', '.join(st)
    
        display(self,equation,str1)      
    except:
        print("Error try again")
        display(self)    


class Equation_Solver(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        drawing_widget = DrawingWidget()

        self.equation_label = Label(text='Your Equation is: ',size_hint=(1, 0.1),color=(1, 0, 0, 1))
        self.roots_label = Label(text='Result: ',color=(0.3, 0, 1, 1))
        self.equation_label.size_hint_y = 0.05
        self.roots_label.size_hint_y = 0.05
        
        save_button = Button(text='Save', size_hint=(1, 0.1))
        save_button.bind(on_release=self.save_image)
        save_button.background_color = (0.5, 0.5, 0.8, 1) # RGBA 0 to1
        save_button.background_normal = 'gradient1.png' 
        
        predict_button = Button(text='Predict', size_hint=(1, 0.1))
        predict_button.bind(on_release=self.predict)
        predict_button.background_color = (0, 1, 1, 1)
        predict_button.background_normal = 'gradient1.png' 

        reset_button = Button(text='Reset', size_hint=(1, 0.1))
        reset_button.bind(on_release=self.reset_canvas)
        reset_button.background_color = (1, 0, 1, 1)
        reset_button.background_normal = 'gradient1.png' 

        layout.add_widget(drawing_widget)
        layout.add_widget(save_button)
        layout.add_widget(predict_button)
        layout.add_widget(reset_button)
        layout.add_widget(self.equation_label)
        layout.add_widget(self.roots_label)


        return layout

    def save_image(self, instance):
        filename = "can.png"
        widget = self.root.children[5]  # 3

        widget.export_to_png(filename)



    def predict(self, instance):
        find_solution(self)

    def reset_canvas(self, instance):
        self.root.children[5].clear_canvas()

if __name__ == '__main__':
    Equation_Solver().run()


