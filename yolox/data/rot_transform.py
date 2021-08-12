from .data_augment import *

def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def rboxes2points(rbboxes):
    list_points = []
    for rbbox in rbboxes:
        cx,cy,w,h,a = rbbox
        rbox = ((cx,cy), (w, h),a)
        points = cv2.boxPoints(rbox)
        list_points.append(points)
    return np.array(list_points).reshape([-1, 8])

def point_flip_x(points, width):
    rot_xs = points[:,0::2]
    cx = rot_xs.mean()
    new_cx = width - cx
    delta_xs = cx-rot_xs
    new_xs = new_cx + delta_xs
    points[:,0::2] = new_xs
    return points

def _mirror(image, boxes, rot_bboxes):
    _, width, _ = image.shape
    # TODO: train loss model with max angle=180
    # if random.randrange(2):
    #     image = image[:, ::-1]
    #     boxes = boxes.copy()
    #     boxes[:, 0::2] = width - boxes[:, 2::-2]

    #     rot_bboxes[:,0] = width - rot_bboxes[:,0]
    #     rot_bboxes[:,-1] = 180-rot_bboxes[:,-1]

    return image, boxes, rot_bboxes

class RotTrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50, accepted_min_box_size=8):
        self.accepted_min_box_size = accepted_min_box_size
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 10), dtype=np.float32)
            image, r_orig = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_orig = image.copy()
        targets_orig = targets.copy()
        height_orig, width_orig, _ = image_orig.shape
        boxes_orig = targets_orig[:, :4]
        labels_orig = targets_orig[:, 4]
        # rot_bboxes_orig = rboxes2points(targets_orig[:, 5:])
        rot_bboxes_orig = targets_orig[:, 5:]

        boxes_orig = xyxy2cxcywh(boxes_orig)

        image_out = _distort(image)
        image_out, boxes, rot_bboxes = _mirror(image_out, boxes, rot_bboxes_orig)
        height, width, _ = image_out.shape
        image_out, resize_ratio = preproc(image_out, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= resize_ratio
        rot_bboxes[:, :4] *= resize_ratio

        valid_box_ids = np.minimum(boxes[:, 2], boxes[:, 3]) > self.accepted_min_box_size

        boxes_out = boxes[valid_box_ids]
        labels_out = labels[valid_box_ids]
        rot_bboxes_out = rot_bboxes[valid_box_ids]
        

        # if len(boxes_out) == 0:
        #     image_out, r_orig = preproc(image_orig, input_dim, self.means, self.std)
        #     boxes_orig *= r_orig
        #     boxes_out = boxes_orig
        #     labels_out = labels_orig

        labels_out = np.expand_dims(labels_out, 1)

        assert len(labels_out) ==  len(rot_bboxes_out)
        
        targets_out = np.hstack((labels_out, boxes_out, rot_bboxes_out))
        padded_labels = np.zeros((self.max_labels, 10))
        padded_labels[range(len(targets_out))[: self.max_labels]] = targets_out[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_out = np.ascontiguousarray(image_out, dtype=np.float32)
        return image_out, padded_labels
