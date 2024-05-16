import argparse
import json
import os
from pathlib import Path

import cv2
import numpy


def json_to_instance(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    return instance


def load_classes(class_txt_path):
    with open(class_txt_path, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes


class OCRPipeline:
    def __init__(self, onnx_path, input_size, score, iou, class_txt_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.inputs = self.session.get_inputs()[0]
        self.input_size = input_size
        self.score = score
        self.iou = iou
        self.classes = load_classes(class_txt_path)

        shape = (1, 3, self.input_size, self.input_size)
        image = numpy.zeros(shape, dtype='float32')
        for _ in range(10):
            self.session.run(output_names=None, input_feed={self.inputs.name: image})

    def get_class_name(self, class_id):
        if class_id < len(self.classes):
            return self.classes[class_id]
        else:
            return "Unknown"

    def ocr_detect(self, img):
        if img.shape[-1] == 1:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img

        image, scale = self.resize(img_rgb, self.input_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))[::-1]
        image = image.astype('float32') / 255
        image = image[numpy.newaxis, ...]
        outputs = self.session.run(output_names=None, input_feed={self.inputs.name: image})
        outputs = numpy.transpose(numpy.squeeze(outputs[0]))
        boxes = []
        scores = []
        class_indices = []

        for i in range(outputs.shape[0]):
            classes_scores = outputs[i][4:]
            max_score = numpy.amax(classes_scores)
            if max_score >= self.score:
                class_id = numpy.argmax(classes_scores).tolist()
                image, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((image - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)
                class_indices.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.score, self.iou)
        nms_outputs = []
        sorted_indices = sorted(indices.flatten(), key=lambda i: boxes[i][0])

        for i in sorted_indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_indices[i]
            nms_outputs.append([*box, score, class_id])
        return nms_outputs

    @staticmethod
    def resize(image, input_size):
        shape = image.shape

        ratio = float(shape[0]) / shape[1]
        if ratio > 1:
            h = input_size
            w = int(h / ratio)
        else:
            w = input_size
            h = int(w * ratio)
        scale = float(h) / shape[0]

        if len(shape) == 3:  # 三通道图像
            det_image = numpy.zeros((input_size, input_size, shape[2]), dtype=numpy.uint8)
        else:  # 单通道图像
            det_image = numpy.zeros((input_size, input_size), dtype=numpy.uint8)

        resized_image = cv2.resize(image, (w, h))
        det_image[:h, :w] = resized_image

        return det_image, scale


def main(args):
    folder_path = args.input_folder
    output_folder = folder_path + '_out'
    os.makedirs(output_folder, exist_ok=True)
    cfg_path = str(Path(__file__).absolute().parent) + '/cfg_basic.json'
    cfg = json_to_instance(cfg_path)
    score = cfg.get("score_threshold", 0.1)
    iou = cfg.get("iou_threshold", 0.45)
    input_size = cfg.get("input_size", 640)
    image_mode = cfg.get("image_mode", "grayscale")
    #onnx_path = str(Path(__file__).absolute().parent) + '/best.onnx'
    onnx_path = args.modelfile
    class_txt_path = args.classnames
    #class_txt_path = str(Path(__file__).absolute().parent) + '/class_names_list.txt'
    ocr_pipeline = OCRPipeline(onnx_path=onnx_path, input_size=input_size, score=score, iou=iou, class_txt_path=class_txt_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            print("Processing:", filename)
            if image_mode == "grayscale":
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            elif image_mode == "rgb":
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif image_mode == "bgr":
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                raise f"image mode error"
            nms_outputs = ocr_pipeline.ocr_detect(img)
            image = img.copy()
            for output in nms_outputs:
                class_name = ocr_pipeline.get_class_name(output[5])
                pt1 = (output[0], output[1])
                pt2 = (output[0] + output[2], output[1] + output[3])
                cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
                cv2.putText(image, class_name + " " + str(round(output[4], 2)), (int(output[0]), int(output[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR Pipeline')

    parser.add_argument(
        '--input_folder',
        type=str,
        default="/home/yibo/yolov8_train/dataset/ocr_origin_data_det/images/val",
        help='Path to the input image folder')

    parser.add_argument(
        '--modelfile',
        type=str,
        default="/home/adt/ocr_train/ocr_good/n/weights/best.onnx",
        help='Path to the modelfile')
    
    parser.add_argument(
        '--classnames',
        type=str,
        default="/home/adt/ocr_train/ocr_good/n/weights/classlist.txt",
        help='Path to the input classlist')
    args = parser.parse_args()

    main(args)
