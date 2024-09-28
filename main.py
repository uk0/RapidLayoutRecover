import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import os
import json
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull


class UniversalOCRLayout:
    def __init__(self, lang='ch'):
        self.ocr = PaddleOCR(use_angle_cls=True,
                             lang=lang,
                             det_algorithm='DB',
                             det_db_thresh=0.3, # default 0.3
                             det_limit_side_len=768,
                             use_mp=True,
                             total_process_num=6,
                             show_log=True,
                             use_mlu=True,
                             det_db_score_mode='slow',
                             use_dilation=True,
                             )
        self.font_path = '/usr/lib/font/Yuanti.ttc'  # 请替换为适合的字体文件路径
        self.default_font_size = 12

    def process_image(self, image_path, output_path, debug=False):
        # 读取图像
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # 执行OCR
        result = self.ocr.ocr(image_path, cls=True)

        # 处理OCR结果
        text_boxes = self.process_ocr_result(result, width, height)

        # 构建布局
        layout = self.build_layout(text_boxes, width, height)

        # 生成输出
        self.generate_output(layout, output_path, width)

        if debug:
            self.generate_debug_image(image_path, text_boxes, output_path)

    def process_ocr_result(self, result, width, height):
        text_boxes = []
        for line in result:
            for word_info in line:
                box = word_info[0]
                text = word_info[1][0]
                confidence = word_info[1][1]

                # 计算边界框
                polygon = Polygon(box)
                x_min, y_min, x_max, y_max = polygon.bounds

                # 计算文本方向
                orientation = self.calculate_orientation(box)

                text_boxes.append({
                    'text': text,
                    'confidence': confidence,
                    'box': box,
                    'bounds': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'orientation': orientation
                })

        return text_boxes

    def calculate_orientation(self, box):
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if -45 <= angle <= 45 or 135 <= angle <= 180 or -180 <= angle <= -135:
            return 'horizontal'
        elif 45 < angle < 135 or -135 < angle < -45:
            return 'vertical'
        else:
            return 'skewed'

    def build_layout(self, text_boxes, width, height):
        # 使用改进的行分组算法对文本框进行分组
        lines = self.group_into_lines(text_boxes)

        # 检测列
        columns = self.detect_columns(lines)

        # 构建布局结构
        layout = {
            'lines': lines,
            'columns': columns
        }

        return layout

    def group_into_lines(self, text_boxes):
        # 按y坐标排序
        sorted_boxes = sorted(text_boxes, key=lambda x: x['bounds'][1])

        lines = []
        current_line = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            if self.is_same_line(current_line[-1], box):
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda x: x['bounds'][0]))
                current_line = [box]

        if current_line:
            lines.append(sorted(current_line, key=lambda x: x['bounds'][0]))

        return lines

    def is_same_line(self, box1, box2, threshold=0.5):
        _, y1, _, y2 = box1['bounds']
        _, y3, _, y4 = box2['bounds']
        overlap = min(y2, y4) - max(y1, y3)
        height1 = y2 - y1
        height2 = y4 - y3
        return overlap > threshold * min(height1, height2)

    def detect_columns(self, lines):
        if not lines:
            return []

        # 获取所有文本框的左边界
        all_left_bounds = [box['bounds'][0] for line in lines for box in line]

        # 使用聚类算法检测列（这里使用简单的阈值方法，你可以根据需要使用更复杂的聚类算法）
        columns = []
        current_column = [all_left_bounds[0]]
        for bound in all_left_bounds[1:]:
            if abs(bound - current_column[-1]) < 20:  # 阈值可以根据需要调整
                current_column.append(bound)
            else:
                columns.append(int(sum(current_column) / len(current_column)))
                current_column = [bound]

        if current_column:
            columns.append(int(sum(current_column) / len(current_column)))

        return sorted(columns)

    def generate_output(self, layout, output_path, image_width):
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in layout['lines']:
                formatted_line = self.format_line(line, layout['columns'], image_width)
                f.write(formatted_line + '\n')

    def format_line(self, line, columns, image_width):
        formatted_line = ''
        last_end = 0
        for box in line:
            start = box['bounds'][0]
            column_index = self.find_column_index(start, columns)
            if column_index is not None:
                spaces = ' ' * max(0, int((columns[column_index] - last_end) / 5))
            else:
                spaces = ' ' * max(0, int((start - last_end) / 5))
            formatted_line += spaces + box['text']
            last_end = box['bounds'][2]

        # 添加行末空格以保持对齐
        end_spaces = ' ' * max(0, int((image_width - last_end) / 5))
        formatted_line += end_spaces

        return formatted_line

    def find_column_index(self, x, columns):
        for i, column in enumerate(columns):
            if abs(x - column) < 25:  # 阈值可以根据需要调整
                return i
        return None

    def generate_debug_image(self, image_path, text_boxes, output_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(self.font_path, self.default_font_size)

        for box in text_boxes:
            points = box['box']
            draw.polygon([tuple(point) for point in points], outline=(255, 0, 0))
            draw.text((points[0][0], points[0][1]), box['text'], font=font, fill=(0, 255, 0))

        debug_image_path = os.path.splitext(output_path)[0] + '_debug.png'
        image.save(debug_image_path)
        print(f"Debug image saved to {debug_image_path}")


# 使用示例
ocr_layout = UniversalOCRLayout()
image_path = 'img_1.png'  # 替换为你的图像路径
output_path = 'ocr_result.txt'  # 输出文件路径

ocr_layout.process_image(image_path, output_path, debug=True)
