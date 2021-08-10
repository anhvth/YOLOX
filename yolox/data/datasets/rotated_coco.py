from numpy import dtype, logical_and

from .coco import *
import cv2



def coco_segmentation_to_rbox(segmentation):
    x = segmentation
    x = np.array(x).reshape([-1,2]).astype(int)
    return cv2.minAreaRect(x)

class RotatedCOCODataset(COCODataset):
    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 10))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            (cx, cy), (w,h), a = coco_segmentation_to_rbox(obj['segmentation'])
            assert a>=0 and a <= 90, a
            res[ix, 5:] = (cx, cy, w, h, a)

        img_info = (height, width)

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"

        del im_ann, annotations

        return (res, img_info, file_name)


    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )

        img = cv2.imread(img_file)
        assert img is not None
        assert np.logical_and(res[:,-1] >=0 , res[:,-1]<=90).all()
        return img, res.copy(), img_info, np.array([id_])