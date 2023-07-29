#!/usr/bin/env python3

import cv2, imutils
import numpy as np
import sys, math, logging
from os.path import basename, splitext, exists, isdir

log = logging.getLogger()
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

if len(sys.argv) == 1:
  log.info("No output directory was specified");
  sys.exit(0)

if len(sys.argv) == 2:
  log.info("No files were specified as arguments");
  sys.exit(0)

draw_size = 1024

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', draw_size, draw_size)
cv2.moveWindow('image', 0, 0);

thresh = 210
blur = 5

out_path = sys.argv[1]
if not exists(out_path):
  log.info("The specified output path does not exist: %s", out_path)
  sys.exit(0)

if not isdir(out_path):
  log.info("The specified output path is not a directory: %s", out_path)
  sys.exit(0)


path = None
canvas = None
draw_canvas = None
draw_scale = 1

# 0 = normal, 1 = operating channel, 2 = thresh
view_mode_str = ['NORMAL', 'OPERATING', 'THRESH']
view_mode = 0

# 0 = bgr2gray, 1=blue, 2=green, 3=red
auto_channel_str = ['GRAY', 'BLUE', 'GREEN', 'RED']
auto_channel = 0

points = []
active_point = None
rects = []

editing_rect_idx = -1
release_editing = False
set_editing_points = False

def load_image():
  img = cv2.imread(path)
  log.info("Loading %s", path)
  canvas_size = round(img.shape[0] * 1.2) if img.shape[0] > img.shape[1] else round(img.shape[1] * 1.2)
  global canvas
  canvas = np.zeros((canvas_size, canvas_size, 3), np.uint8)
  canvas[:] = (255, 255, 255)

  global draw_scale
  draw_scale = canvas_size * 1.0 / draw_size
  left_offset = round((canvas_size - img.shape[0])/2)
  top_offset = round((canvas_size - img.shape[1])/2)
  canvas[left_offset : left_offset + img.shape[0], top_offset : top_offset + img.shape[1]] = img
  global draw_canvas
  global draw_canvas_gray
  draw_canvas = cv2.resize(canvas, (draw_size, draw_size))
  re_guess_rects()

def auto_channel_img():
  if auto_channel == 0:
    return cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
  elif auto_channel == 1:
    return draw_canvas[:,:,0]
  elif auto_channel == 2:
    return draw_canvas[:,:,1]
  elif auto_channel == 3:
    return draw_canvas[:,:,2]

def gen_thresh():
  img = auto_channel_img()
  ret, result = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
  kernel = np.ones((5,5),np.uint8)
  result = cv2.dilate(result, kernel, iterations = 1)
  result = cv2.erode(result, kernel, iterations = 1)
  if blur > 0:
     #result = cv2.GaussianBlur(result, (blur,blur), 0)
    result = cv2.medianBlur(result, blur)
    ret, result = cv2.threshold(result, thresh, 255, cv2.THRESH_BINARY)
  #kernel = cv2.getStructuringElement()
  #kernel = np.ones((5,5),np.uint8)
  #result = cv2.dilate(result, kernel, iterations = 1)
  #result = cv2.erode(result, kernel, iterations = 1)
  #result = cv2.dilate(result, kernel, iterations = 1)
  return result

def adjust_guess(t, b):
  global thresh
  global blur
  if t > 255:
    t = 255
  elif t < 0:
    t = 0
  if b < 0:
    b = 0
  elif b == 2:
    b = 1
  thresh = t
  blur = b
  re_guess_rects()

def re_guess_rects():
  del points[:]
  del rects[:]
  guess_rects()
  render()

def guess_rects():
  log.info("guess_rects thresh=%d, blur=%d", thresh, blur)
  result = gen_thresh()
  contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    area = cv2.contourArea(cnt)
    # sane-ish size
    if area < draw_canvas.shape[0] * draw_canvas.shape[1] * .9 and area > 100 * 100:
      rect = cv2.minAreaRect(cnt)
      center, size, theta = rect
      t = math.fabs(theta)
      # sane rotation
      if t < 20 or t > 70:
        # sufficient width and height
        if size[0] > draw_canvas.shape[0] * 0.05 and size[1] > draw_canvas.shape[1] * 0.05:
          # sane aspect ratio
          if size[0] / size[1] < 3 and size[1] / size[0] < 3:
            rects.append(rect)

  #cv2.drawContours(draw_canvas, contours, -1, (0,255,0), 3)
  #cv2.imshow('thresh',thresh)

def put_line(img, line, text):
  y = 18 + 25 * line
  cv2.putText(img, text, (0, y), cv2.FONT_HERSHEY_TRIPLEX, .7, (0,0,0), 1, cv2.FILLED)

def render():
  global set_editing_points
  if view_mode == 0:
    img = draw_canvas.copy()
  elif view_mode == 1:
    # show thresh operating canvas
    t = auto_channel_img()
    img = cv2.merge((t,t,t))
  elif view_mode == 2:
    # show thresh as b/w
    t = gen_thresh()
    img = cv2.merge((t,t,t))

  for p in points:
    color = (0,255,0)
    if editing_rect_idx > -1:
      color = (240, 24, 38)
    cv2.circle(img, p, 6, color=color, thickness=2, lineType=8, shift=0)

  put_line(img, 0, 'File %d/%d: %s' % (file_idx, len(sys.argv)-1, basename(path)))
  put_line(img, 1, 'Threshold: %d' % thresh)
  put_line(img, 2, 'Blur: %d' % blur)
  put_line(img, 3, 'View: %s' % view_mode_str[view_mode])
  put_line(img, 4, 'Channel: %s' % auto_channel_str[auto_channel])

  draw_rects = rects[:]

  # recalculate bounding rect for the selected editing rect
  if editing_rect_idx > -1:
    newRect = cv2.minAreaRect(np.int0(points))
    draw_rects[editing_rect_idx] = newRect
    rects[editing_rect_idx] = newRect
  else:
    if active_point:
      log.info("active_point %s", active_point)
    if active_point:
      cv2.circle(img, active_point, 6, color=(200,100,0), thickness=2, lineType=8, shift=0)
      if len(points) == 3:
        # use activepoint to draw a rect
        pts = points[:] + [active_point]
        draw_rects.append(cv2.minAreaRect(np.int0(pts)))
      
  for i, r in enumerate(draw_rects):
    box = cv2.boxPoints(r)
    box = np.int0(box)
    color = (100, 100, 255)
    if i == editing_rect_idx:
      color = (56, 240, 24)
    cv2.drawContours(img, [box], 0, color, 2)
  cv2.imshow('image', img)
  

def reset_selected_rect():
  # log.info('Resetting selected rect')
  global editing_rect_idx
  global release_editing
  editing_rect_idx = -1
  # clear points if they were previously rendered for a selected rect
  if set_editing_points:
    points.clear()
  release_editing = True

def canvas_click(event, x, y, flags, param):
  global active_point
  global editing_rect_idx
  global release_editing
  global points
  global set_editing_points
  # select a rect to edit
  if event == cv2.EVENT_MBUTTONDOWN:
    log.info("Mousewheel-click: %d,%d", x, y)    
    active_point = None
    for i, r in enumerate(rects):
      box = cv2.boxPoints(r)
      box = np.int0(box)
      test = cv2.pointPolygonTest(box,(x,y),True)
      if test > -1:
        log.info('Selected rect #%d', i)
        set_editing_points = False
        release_editing = False
        editing_rect_idx = i
        points.clear()
          # if editing_rect_idx > -1 and not set_editing_points:
        box = cv2.boxPoints(rects[editing_rect_idx])
        box = np.int0(box)
        for p in box:
          points.append(p)
        set_editing_points = True

        render()
        return
    reset_selected_rect()
    render()

  # edit the points of a selected rect, 
  # or adjust the position of the last point in a new rect while the 
  # left mouse button is held down
  if event == cv2.EVENT_MOUSEMOVE and flags == 1:
    log.info("Down: %d,%d flags=%s", x, y, flags)    
    if len(points) == 4:
      # replace existing point
      closest_idx = None
      closest_dist = None
      for i in range(0, len(points)):
        dx = math.fabs(points[i][0] - x)
        dy = math.fabs(points[i][1] - y)
        dist = math.sqrt(dx*dx + dy*dy)
        if closest_idx is None or dist < closest_dist:
          closest_idx = i
          closest_dist = dist
      log.info("Replace closest point at dist %f", closest_dist)
      if editing_rect_idx == -1:
        points.pop(closest_idx)
      else:
        if closest_idx is not None:
          points[closest_idx] = (x, y)
    active_point = (x,y)
    render()
  # set the last point for a new rect
  elif event == cv2.EVENT_LBUTTONUP and editing_rect_idx == -1:
    log.info("Up: %d,%d", x, y)
    active_point = None
    p = (x,y)
    if len(points) < 4:
      points.append(p)
    render()
   

cv2.setMouseCallback('image', canvas_click)


def rotate(img, center, angle):
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  return cv2.warpAffine(img, matrix, img.shape[1::-1])

def do_crops():
  idx = 0
  for r in rects:
    idx = idx + 1
    name = "%s_%d.jpg" % (splitext(basename(path))[0], idx)
    log.info("name %s", name)
    # tan = opp / agj
    center, size, theta = r
    scaled_center = (draw_scale * center[0], draw_scale * center[1])
    scaled_size = (math.ceil(draw_scale * size[0]), math.ceil(draw_scale * size[1]))
    log.info("angle %f", theta)
    rotated = rotate(canvas, scaled_center, theta)
    cropped = cv2.getRectSubPix(rotated, scaled_size, scaled_center)
    #cv2.imshow('image', cropped)
    cv2.imwrite(out_path + "/" + name, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

file_idx = 2

def load_next():
  del points[:]
  del rects[:]
  log.info("Loading %d of %d", file_idx, len(sys.argv)-1)
  global path
  path = sys.argv[file_idx]
  load_image()

load_next()

while 1:
  k = cv2.waitKey(0)
  if k == 27 or k == 113:
    # escape or q
    sys.exit(0)
  elif k == 119:
    # up with w
    adjust_guess(thresh+2, blur)
  elif k == 115:
    # down with s
    adjust_guess(thresh-2, blur)
  elif k == 97:
    # less blur with a
    adjust_guess(thresh, blur - 2)
  elif k == 100:
    # more blur with d
    adjust_guess(thresh, blur + 2)
  elif k == 118:
    # v
    view_mode = (view_mode + 1) % 3
    render()
  elif k == 99:
    # c
    auto_channel = (auto_channel + 1) % 4
    re_guess_rects()
  elif k == 13:
    # enter
    log.info("enter")
    if len(rects) > 0 and len(points) == 0:
      do_crops()
      file_idx = file_idx + 1
      if file_idx >= len(sys.argv):
        break
      load_next()
    else:
      log.info("Ignoring enter because len(rects)=%d and len(points)=%d", len(rects), len(points))
  elif k == 98:
    # 'b' for back
    file_idx = file_idx - 1
    if file_idx == 0:
      file_idx = 1
    load_next()
  elif k == 110:
    # 'n' for next
    file_idx = file_idx + 1
    if file_idx >= len(sys.argv):
      break
    load_next()
  elif k == 8:
    # backspace
    if len(points) > 0:
      points.pop()
    elif len(rects) > 0:
      rects.pop()
    render()
  elif k == 32:
    # spacebar
    if editing_rect_idx <= -1:
      if len(points) == 4: 
        rect = cv2.minAreaRect(np.int0(points))
        center, size, theta = rect
        log.info("center: %f,%f  -  size %f,%f  -  theta %f", center[0], center[1], size[0], size[1], theta)
        rects.append(rect)
        del points[:]
        render()
      else:
        log.info("Can't finalize, only %d points set", len(points))
    else:
      reset_selected_rect()
      render()
    

log.info("DONE")
cv2.destroyAllWindows()