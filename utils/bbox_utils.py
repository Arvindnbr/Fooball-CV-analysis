

def get_bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width_height(bbox):
    x1,y1,x2,y2 = bbox
    return int(x2-x1), int(y2-y1) 

def get_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)