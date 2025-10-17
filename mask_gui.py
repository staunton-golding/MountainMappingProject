import os
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from collections import deque
import webcolors
import numpy as np
from pathlib import Path

#I made this gui to refine masks segmented with GroundingSAM. To use, change image_dir, mask_dir and mask_already_done_dir. Also change output paths
color_name = "white"
image_num = 1

image_dir = "./training_data/images"
mask_dir = "./training_data/labels"
mask_already_done_dir = "./training_images_masks"

#get image and mask files, + masks yet to be corrected
image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg') or f.endswith('.png')]
mask_files_already_done = [f for f in (os.listdir(mask_already_done_dir)) if f.endswith('.jpg')]
mask_files_already_done_check = [item for item in (sorted([item.partition('_')[0] for item in mask_files_already_done]))]
image_files_check = [item for item in (sorted([item.partition('.')[0] for item in image_files]))]
still_to_do = set(image_files_check) - set(mask_files_already_done_check)

mask_files = [(item + "_mask.jpg")  for item in still_to_do]
image_files = [(item + ".jpg") for item in still_to_do]

line_width = 4

num_images = len(image_files)
num_masks = len(mask_files)

#print number of images and masks still to be done
print(f"num images = {num_images}, num_masks = {num_masks}")

#output paths
out_label_path = mask_already_done_dir
out_image_path = "./resized_images"

if os.path.isdir(out_label_path) == 0:
    os.mkdir(out_label_path)

if os.path.isdir(out_image_path) == 0:
    os.mkdir(out_image_path)

class MaskPaint:
    def __init__(self, master):
        self.master = master
        master.title("Mask Painter GUI")
        
        self.canvas_width = 1024
        self.canvas_height = 1024

        ##note mask is shown black over image, but pixel is white - white paint over image equates to mask
        #however, when you fill an area, the colors are reversed (white becomes black)
        #don't try to fill color if pixel is already fill color color, restart gui if it stalls
        #if you make the wrong move / wrong color, left then right resets canvas

        ##LIST OF BINDINGS
        #hold and move mouse to paint, release mouse to stop paint
        #double click to fill image
        #enter to save mask
        #left for next image
        #right to previous image
        #c to switch colors
        #shift up to increase brush size
        #shift down to increase brush size

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_paint)
        self.canvas.bind("<Double-Button-1>", self.start_fill)
        self.current_color = color_name

        #internal bound buttons (keyboard events not registering until implementing this)
        self.right_button = tk.Button(master, text = '>')
        self.right_button.bind('<Button-1>', self.image_right)
        self.master.bind('<Right>', self.image_right)

        self.left_button = tk.Button(master, text = '>')
        self.left_button.bind('<Button-1>', self.image_left)
        self.master.bind('<Left>', self.image_left)

        self.save_button = tk.Button(master, text = '>')
        self.save_button.bind('<Button-1>', self.save_image)
        self.master.bind('<Return>', self.save_image)

        self.switch_col_button = tk.Button(master, text = '>')
        self.switch_col_button.bind('<Button-1>', self.switch_color_bin)
        self.master.bind('<c>', self.switch_color_bin)

        self.bigger_brush_button = tk.Button(master, text = '>')
        self.switch_col_button.bind('<Button-1>', self.bigger_brush)
        self.master.bind('<Shift-Up>', self.bigger_brush)

        self.smaller_brush_button = tk.Button(master, text = '>')
        self.smaller_brush_button.bind('<Button-1>', self.smaller_brush)
        self.master.bind('<Shift-Down>', self.smaller_brush)

        self.is_filling = False
        self.last_x, self.last_y = None, None
        
        self.clear_canvas()

    #decrease brush size
    def smaller_brush(self, event):
        global line_width
        if line_width == 1:
            print("Brush already at smallest")
        else:
            line_width -=1
            print(f"Brush is now {line_width} pixels")

    #increase brush size
    def bigger_brush(self, event):
        global line_width
        if line_width > 50 :
            print("Brush already at biggest (50)")
        else:
            line_width +=1
            print(f"Brush is now {line_width} pixels")

    #reset canvas (with new image if left or right pressed)
    def clear_canvas(self):
        self.canvas.delete("all")        
        
        image_to_open = os.path.join(image_dir, image_files[image_num])
        mask_to_open = os.path.join(mask_dir, mask_files[image_num])

        try:  
            self.bg_image_pil = Image.open(image_to_open)
            self.bg_image_pil = self.bg_image_pil.resize((1024, 1024), Image.LANCZOS) # Use LANCZOS for better quality
            self.bg_image_tk = ImageTk.PhotoImage(self.bg_image_pil)

        except FileNotFoundError:
            print("Background image not found. Please provide a valid path.")
            self.bg_image_tk = None

        # If a background image was loaded, display it on the canvas
        if self.bg_image_tk:
            #backgroundLabel = tk.Label(self.canvas,image=self.bg_image_tk)
            #backgroundLabel.place(x=0,y=0)#,relx=1,rely=1)
            print("loading image")
            self.canvas.create_image(0, 0, image=self.bg_image_tk, anchor=tk.NW)
            
        print(f"image opened is {image_to_open}")
        
        pil_image = Image.open(mask_to_open)
        
        pil_image = pil_image.resize((1024, 1024), Image.BOX)
        np_mask = np.array(pil_image)
        np_mask[np_mask>0] = 255
        im = Image.fromarray(np_mask.astype(np.uint8))
        im.save('temp.bmp')
        self.image = Image.open('temp.bmp').convert(mode = "1")
        os.remove('temp.bmp')
        self.tk_image = ImageTk.BitmapImage(self.image)
        self.draw = ImageDraw.Draw(self.image)
        
        self.image_item = self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)

    #switch color from black to white or vise versa
    def switch_color_bin(self, event):
        global color_name
        
        if color_name == "white":
            color_name = "black"
            
        else:
            color_name = "white"
        
        print(f"color is now {color_name}")
        self.current_color = color_name

    #save mask
    def save_image(self, event):

        output_file_mask_path = out_label_path+"/"+Path(image_files[image_num]).stem+"_mask.jpg"
        output_file_image_path = out_image_path+"/"+Path(image_files[image_num]).stem+"_resized.jpg"
        print(output_file_image_path)
        np_photo = np.array(self.image)
        #np_photo = np_photo[:,:,0]
        mask_image = Image.fromarray(np_photo)
        mask_image.save(output_file_mask_path)
        self.bg_image_pil.save(output_file_image_path)
        print(f"Mask saved to {output_file_mask_path}, image saved to {output_file_image_path}")
        
    #next image
    def image_right(self, event):
        global image_num 
        print(f"image number {image_num}")       
        if image_num < num_images:
            image_num += 1
            self.clear_canvas()
        else:
            print("already at last image")

    #previous image
    def image_left(self, event):
        global image_num
        print(f"image number {image_num}")        
        if image_num > 0:
            image_num -= 1
            self.clear_canvas()
        else:
            print("already at first image")

    def start_paint(self, event):
        self.last_x, self.last_y = event.x, event.y

    def stop_paint(self, event):
        self.last_x, self.last_y = None, None

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y, 
                fill=self.current_color, width=line_width, capstyle=tk.ROUND, smooth=tk.TRUE
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=self.current_color, width=line_width
            )
        self.last_x, self.last_y = event.x, event.y

    def start_fill(self, event):
        x, y = event.x, event.y
        
        if self.image.getpixel((x, y)) != self.current_color:
            self.flood_fill(x, y, self.image.getpixel((x, y)), self.current_color)
        else:
            print("The target and fill color are the same")

    def flood_fill(self, x, y, target_color, fill_color):
        
        rgb_tuple = webcolors.name_to_rgb(color_name)
        # Convert RGB tuple to a single 24-bit integer
        integer_color = ((rgb_tuple[0] << 16), (rgb_tuple[1] << 8), rgb_tuple[2])

        if target_color == integer_color:
            print("target and fill color the same, not doing anything")
            return

        queue = deque([(x, y)])
        self.draw.point((x,y), fill=fill_color)

        while queue:
            cx, cy = queue.popleft()
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < self.canvas_width and 0 <= ny < self.canvas_height:
                    if self.image.getpixel((nx, ny)) == target_color:
                        self.draw.point((nx, ny), fill=fill_color)
                        queue.append((nx, ny))
        
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_item, image=self.tk_image)

root = tk.Tk()
app = MaskPaint(root)
root.mainloop()