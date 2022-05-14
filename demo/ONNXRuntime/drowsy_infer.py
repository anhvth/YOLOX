
import cv2
import torch
import numpy as np
# from cython_bbox import bbox_overlaps as bbox_ious

def adjust_bb(bbox, h,w):
    """
        Make sure bbox is inside image
    """
    x1,y1,x2,y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return x1,y1,x2,y2


def bb_intersection_over_b_area(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxBArea)

    # return the intersection over union value
    return iou

class Face:
    def __init__(self, face, left_eye, right_eye, mouth):
        self.face = face
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.mouth = mouth
        
    def draw_rect(self, img, box, thickness=1, color=(0,255,0)):
        if box is None:
            return img
        fx1,fy1,fx2,fy2 = [int(_) for _ in self.face[:4]]
        
        x1,y1,x2,y2 = [int(_) for _ in box[:4]]
        x1 -= fx1
        x2 -= fx1
        y1 -= fy1
        y2 -= fy1
        
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
        return img
    
    def vis(self, mask=None):
        x1,y1,x2,y2 = [int(_) for _ in self.face[:4]]
        h,w = y2-y1, x2-x1
        if mask is None:
            mask = np.zeros([h,w, 3]).astype('uint8')
        else:
            mask = mmcv.imresize(mask, (h,w))
        mask = self.draw_rect(mask, self.left_eye, color=(255,0))
        mask = self.draw_rect(mask, self.right_eye)
        mask = self.draw_rect(mask, self.mouth)
        return mask
        # cv2.rectangle()
        
    def is_valid(self):
        for x in [self.right_eye, self.left_eye, self.mouth]:
            if x is not None: return True
        return False
        
class FaceAnalizer:
    def __init__(self, dets, max_face=1):
        self.dets
        self.max_face = max_face
        
    
    def get_face(self):
        face_bbox = self.get_bbox(lambda c:c==0)
        if len(face_bbox) == 0:
            return None
        return face_bbox[0]
    
    
    def get_mouth_eye(self, face, over_lap_thr=0.5):
    
        other_bboxes = self.get_bbox(lambda c:c!=0)
        if len(other_bboxes) == 0:
            return dict(eyes=[], mouth=[])
        
        eyes = []
        mouths = []
        for other_bbox in other_bboxes:
            if other_bbox[-1] == 1 and bb_intersection_over_b_area(face, other_bbox) > over_lap_thr:
                eyes.append(other_bbox)
            elif other_bbox[-1] == 2 and bb_intersection_over_b_area(face, other_bbox) > over_lap_thr:
                mouths.append(other_bbox)
                
        return dict(eyes=np.array(eyes), mouth=np.array(mouths))
        
        
    
    def get_bbox(self,  cat_lambda, score_thr=0.5):
        anns = [ann for ann in self.anns if  cat_lambda(ann['category_id']) and ann['score']>score_thr]
        x = []
        for ann in anns:
            x.append([*ann['bbox'], ann['score'], ann['category_id']])
        if len(x) == 0:
            return x
        
        x = list(sorted(x, key=lambda b:b[-2]))
        x = np.array(x)
        x[:,2] += x[:,0]
        x[:,3] += x[:,1]
        
        return x
    def get_prominent_face(self):
        face = self.get_face()
        if face is None:
            # print('No face found', img_id)
            return None
        
        em = self.get_mouth_eye(face)
        em.update(dict(face=face))
        mouth = em['mouth']
        if len(mouth) == 0:
            max_mouth = [None]
        else:
            max_mouth = mouth[mouth[:,-2]==mouth[:,-2].max()]
        
        face = em['face']
        left_eye, right_eye = None, None
        for eye in em['eyes']:
            x1,y1,x2,y2 = eye[:4]
            cx = (x1+x2)/2
            if abs(cx - face[0]) < abs(cx - face[2]):
                left_eye = eye
            else:
                right_eye = eye
        return Face(face, left_eye, right_eye, max_mouth[0])
    



def get_face(anns):
    face = FaceAnalizer(anns).get_prominent_face()
    return face
    
def per_img_preproc(img, le, re, mo, size=32):
    from timm.data.transforms_factory import  transforms_imagenet_eval
    def crop(img, bbox, size=None):
        try:
            if bbox is None:
                # h,w = size
                return np.zeros([*size, 3]).astype('uint8')
            x,y,x2,y2 = [int(_) for _ in bbox[:4]]
            img = mmcv.imread(img)
            img = img[y:y2, x:x2]
            if size is not None:
                img = mmcv.imrescale(img, size[::-1])
                img = mmcv.impad(img, shape=size)
            return img
        except:
            return np.zeros([*size, 3]).astype('uint8')

    left_eye = crop(img, le, size)
    right_eye = crop(img, re, size)
    mouth = crop(img, mo, size)
    x =[left_eye, right_eye, mouth]

    x = [image_transform(Image.fromarray(_)) for _ in x]
    lrm = torch.stack(x)
    return lrm
    
