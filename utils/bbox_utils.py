

def get_bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width_height(bbox):
    x1,y1,x2,y2 = bbox
    return int(x2-x1), int(y2-y1) 