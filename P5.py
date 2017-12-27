import pickle
import collections
import cv2

from moviepy.editor import VideoFileClip

from classify import *

##### load pickle with classifier and training parameters

dist_pickle = pickle.load( open("p5_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
colorspace = dist_pickle["colorspace"]
hog_channel = dist_pickle["hog_channel"]
image_format = dist_pickle["image_format"]

print("orient = {}, pix_per_cell = {}, cells_per_block = {}, spatial_size = {}, hist_bins = {}, colorspace = {}".format(
    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace))
print("hog channel = {}, image_format = {}".format(hog_channel, image_format))

##### save bounding boxes for last 4 frames

bbox_buffer = collections.deque(maxlen=4) 

################################################################################

def process_image(image):
    """image processing pipeline"""

    ystart = [368, 400, 400]
    ystop = [560, 592, 656]
    scale = [1.0, 1.5, 2.0]

    bbox_list = find_multiscale_cars(image, ystart, ystop, scale, svc, X_scaler,
                                     orient, pix_per_cell, cell_per_block,
                                     spatial_size, hist_bins, colorspace)
    
    bbox_buffer.append(bbox_list)
    labeled_image = heat_map(image, bbox_buffer)

    return labeled_image
################################################################################

# load video, process each frame, write processed video
input_video = VideoFileClip("project_video.mp4")#.subclip(20, 25)
output_video = input_video.fl_image(process_image)
output_video.write_videofile('test_output.mp4', audio=False)
