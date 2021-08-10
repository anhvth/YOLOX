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
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 10), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # rot_bboxes_o = rboxes2points(targets_o[:, 5:])
        rot_bboxes_o = targets_o[:, 5:]

        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = _distort(image)
        image_t, boxes, rot_bboxes = _mirror(image_t, boxes, rot_bboxes_o)
        height, width, _ = image_t.shape
        image_t, resize_ratio = preproc(image_t, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= resize_ratio
        rot_bboxes[:, :4] *= resize_ratio

        valid_box_ids = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        boxes_t = boxes[valid_box_ids]
        labels_t = labels[valid_box_ids]
        rot_bboxes_t = rot_bboxes[valid_box_ids]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t, rot_bboxes_t))
        padded_labels = np.zeros((self.max_labels, 10))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels
