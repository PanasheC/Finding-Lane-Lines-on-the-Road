
# Finding Lane Lines in Road 
In this project, I used Python and OpenCV to find lane lines in a road video stream. 

**My approach is as follows**:

 * Identifying Lane Lines by Color Selection
 * Gaussian Blur to reduce noise
 * Region Masking or Region of Interest Selection
 * Canny Edge Detection
 * Hough Transform to find Lane Lines

I applied all the above techniques to process video clips to find lane lines in them using MoviePy which is a Python module for video editing, which can be used for basic operations (like cuts, concatenations, title insertions).

# Project Specification 

## Lane Finding pipeline

| Criteria | Meets Specification |
| -------- | ------------------- |
| Does the pipeline for the line identification take road images from the video as input and return an annotated video stream as output? | The output video is an annotated version of the input video |
|Has a pipeline been implemented that uses the helper functions and / or other code to roughly identify the left and right lanes lines with either line segments or solid lines?  |In a rough sense, the left and right lane lines are accurately annotated through almost all of the video. Annotations can be segmented or solid lines. |


# The Code

## We import the necessary python libraries


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, ImageClip
from IPython.display import HTML
%matplotlib inline
```

# Helper Methods for the pipeline
We create helper functions for the pipeline by wrapping OpenCV Methods

## Gray Scaling

The images should be converted into gray scaled ones in order to detect shapes (edges) in the images.  This is because the Canny edge detection measures the magnitude of pixel intensity changes or gradients.


```python
def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

## Canny Edge Detection

The Canny edge detector was developed by John F. Canny.  

We want to detect edges in order to find straight lines especially lane lines.  For this, 

-  `cv2.cvtColor` to convert images into gray scale
-  `cv2.GaussianBlur` to smooth out rough edges 
-  `cv2.Canny` to find edges


```python
def canny_edge(img, low_threshold, high_threshold):
    """ The Canny edge detector is an edge detection operator that uses a
    multi-stage algorithm to detect a wide range of edges in images. """
    return cv2.Canny(img, low_threshold, high_threshold)
```

## Gaussian Blur

When there is an edge (i.e. a line), the pixel intensity changes rapidly (i.e. from 0 to 255) The GaussianBlur takes a kernel_size parameter which you'll need to play with to find one that works best. I tried 3, 5, 9, 11, 15, 17 (they must be positive and odd) and check the edge detection result. The bigger the kernel_size value is, the more blurry the image becomes. 


```python
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

Both the white and yellow lines are clearly recognizable. We build a filter to select the white and yellow lines. we need to select particular range of each channels (Hue, Saturation and Light). For this,

- cv2.inRange to filter the white color and the yellow color seperately.
- function returns 255 when the filter conditon is satisfied. Otherwise, it returns 0.
- cv2.bitwise_or to combine these two binary masks.
    The combined mask returns 255 when either white or yellow color is detected.
- cv2.bitwise_and to apply the combined mask onto the original RGB image



When finding lane lines, we don't need views that are out of context. We are interested in the areas surrounded by the lines only. This is our region of interest.


```python
def region_of_interest(img, vertices):
    """ Applies an image mask.Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color 
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```


```python
def draw_lines(img, lines, color=(255, 0, 0), thickness=10):
    """Iterate over the output "lines" and draw lines on the blank"""
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
```

## Hough Transform Line Detection

I'm using `cv2.HoughLinesP` to detect lines in the edge images.

There are several parameters you'll need tweaking:

- rho – Distance resolution of the accumulator in pixels.
- theta – Angle resolution of the accumulator in radians.
- threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes (> `threshold`).
- minLineLength – Minimum line length. Line segments shorter than that are rejected.
- maxLineGap – Maximum allowed gap between points on the same line to link them.


```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```


```python
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Averaging and Extrapolating Lines

There are multiple lines detected for a lane line.  We should come up with an averaged line for that.

Also, some lane lines are only partially recognized.  We should extrapolate the line to cover full lane line length.

We want two lane lines: one for the left and the other for the right.  The left lane should have a positive slope, and the right lane should have a negative slope.  Therefore, we'll collect positive slope lines and negative slope lines separately and take averages.

**in the image, `y` coordinate is reversed.  The higher `y` value is actually lower in the image.  Therefore, the slope is negative for the left lane, and the slope is positive for the right lane.**


```python
def separate_lines(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0: 
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])
    
    return right, left
```


```python
def extend_point(x1, y1, x2, y2, length):
    """ Takes line endpoints and extroplates new endpoint by a specfic length"""
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) 
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y
```


```python
def reject_outliers(data, cutoff, thresh=0.08):
    """Reduces jitter by rejecting lines based on a hard cutoff range and outlier slope """
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    return data[(data[:, 4] <= m+thresh) & (data[:, 4] >= m-thresh)]
```


```python
def merge_lines(lines):
    """Merges all Hough lines by the mean of each endpoint, 
       then extends them off across the image"""
    
    lines = np.array(lines)[:, :4] ## Drop last column (slope)
    
    x1,y1,x2,y2 = np.mean(lines, axis=0)
    x1e, y1e = extend_point(x1,y1,x2,y2, -1000) # bottom point
    x2e, y2e = extend_point(x1,y1,x2,y2, 1000)  # top point
    line = np.array([[x1e,y1e,x2e,y2e]])
    
    return np.array([line], dtype=np.int32)
```

## Stabilizing for shaded areas
We create a color mask that will highlights the whites and yellows in the frame. This will ensure the Hough lines are more easily detected in shaded regions or low contrast regions. 


```python
def merge_prev(line, prev):
    """ Extra Challenge: Reduces jitter and missed lines by averaging previous 
        frame line with current frame line. """
    if prev != None:
        line = np.concatenate((line[0], prev[0]))
        x1,y1,x2,y2 = np.mean(line, axis=0)
        line = np.array([[[x1,y1,x2,y2]]], dtype=np.int32)
        return line
    else: 
        return line
```

## Create the pipeline 
We add a global variable for the line from the prior frame. This will be averaged with the current frame to  prevent jittery line detection on the video footage.  


```python
global right_prev
global left_prev
right_prev = None
left_prev = None
```


```python
def pipeline(image, preview=False):
    global right_prev
    global left_prev
    bot_left = [250, 660]
    bot_right = [1100, 660]
    apex_right = [725, 440]
    apex_left = [580, 440]
    v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]
    
    ### Added a color mask to deal with shaded region
    color_low = np.array([187,187,0])
    color_high = np.array([255,255,255])
    color_mask = cv2.inRange(image, color_low, color_high)
    
    gray = gray_scale(image)
    blur = gaussian_blur(gray, 3)
    blur = weighted_img(blur, color_mask)

    edge = canny_edge(blur, 100, 300)
    mask = region_of_interest(edge, v)

    lines = cv2.HoughLinesP(mask, 0.5, np.pi/180, 10, np.array([]), minLineLength=90, maxLineGap=200)

    right_lines, left_lines = separate_lines(lines)

    right = reject_outliers(right_lines, cutoff=(0.45, 0.75))
    right = merge_lines(right)
    right = merge_prev(right, right_prev)
    right_prev = right

    left = reject_outliers(left_lines, cutoff=(-1.1, -0.2))
    left = merge_lines(left)
    left = merge_prev(left, left_prev)
    left_prev = left
    
    lines = np.concatenate((right, left))
    line_img = np.copy((image)*0)
    draw_lines(line_img, lines, thickness=10)
    
    line_img = region_of_interest(line_img, v)
    final = weighted_img(line_img, image)
    
    return final

```

## Process the image pipeline


```python
def process_image(image):
    result = pipeline(image)
    return result
```


```python
new_clip_output = 'detectlanes.mp4'
clip1 = VideoFileClip("test.mp4")
new_clip = clip1.fl_image(process_image)
%time new_clip.write_videofile(new_clip_output, audio=False)
```

    [MoviePy] >>>> Building video detectlanes.mp4
    [MoviePy] Writing video detectlanes.mp4


    100%|██████████| 251/251 [00:18<00:00, 12.10it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: detectlanes.mp4 
    
    CPU times: user 4min 23s, sys: 8.19 s, total: 4min 32s
    Wall time: 19.2 s


## Playback the annotated video with lane detection


```python
HTML("""
<video width="640" height="300" controls>
  <source src="{0}" type="video/mp4">
</video>
""".format(new_clip_output))
```





<video width="640" height="300" controls>
  <source src="detectlanes.mp4" type="video/mp4">
</video>




# Conclusion

Using these techniques lane detection works well but does not factor for curvture of the lanes.


```python

```
