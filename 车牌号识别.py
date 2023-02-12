import time
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
import argparse
from rich.console import Console
from tqdm.tk import tqdm
from rich import print

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"


class Timer:

    def __init__(self):
        self.TimeElapsed = 0

    def __enter__(self):
        self.start = time.time()
        print('[cyan]程序开始，正在加载OCR模型...[/cyan]')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.TimeElapsed = self.end - self.start
        print(f'[blink cyan]程序结束，运行时间：{self.TimeElapsed} 秒[/blink cyan]')


def txt_recognize(path, ocr):
    imgs_info = get_plate_imgs(path)
    rec_success, rec_failed, plates_total = 0, 0, 0
    dir_list = os.listdir(path)[1:]
    img_total = len(dir_list)
    for imageName in tqdm(dir_list, desc='正在识别车牌', leave=False, mininterval=0.0001):

        plate_num = next(imgs_info)
        plates_total += plate_num
        for i in range(plate_num):
            img = next(imgs_info)

            """
            cv2.namedWindow(imageName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(imageName, img)
            cv2.waitKey(50)
            cv2.destroyWindow(imageName)
            """
            result = ocr.ocr(img, cls=True)

            try:
                print(f'[green]{result[0][0][1][0]}[/green]')
                rec_success += 1
            except Exception:
                print('[red]识别失败[/red]')
                rec_failed += 1
    return rec_success, rec_failed, img_total, plates_total


def get_plate_imgs(path):
    labels_path = os.path.join(path, 'labels')
    for txtName in os.listdir(labels_path):
        with open(os.path.join(labels_path, txtName)) as f:
            texts = f.readlines()
            yield len(texts)
            for text in texts:
                text = text.replace('\n', ' ').split(' ')
                imgName = txtName.split('.')[0] + '.jpg'
                img_path = os.path.join(path, imgName)
                img = cv2.imread(img_path)
                assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

                x_c, y_c, w, h = float(text[1]), float(text[2]), float(text[3]), float(text[4])
                w, h, x_c, y_c = w * img.shape[1], h * img.shape[0], x_c * img.shape[1], y_c * img.shape[0]
                xmin, xmax, ymin, ymax = x_c - w / 2, x_c + w / 2, y_c - h / 2, y_c + h / 2

                img = Image.fromarray(img)
                img = img.crop((xmin, ymin, xmax, ymax))
                img = img.resize((100, 30), Image.LANCZOS)
                img = np.asarray(img)

                yield img


def main():
    parser = argparse.ArgumentParser('车牌号识别')
    parser.add_argument('--FilePath', type=str, default="./runs/detect", help='Detect结束所得文件夹的地址')
    args = parser.parse_args()

    fileNames = os.listdir(args.FilePath)
    fileNames.sort(key=lambda x: int(x[3:]))
    fileName =fileNames[-1]

    path = os.path.join(args.FilePath, fileName)

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    print('[cyan]OCR模型加载完毕，开始识别...[/cyan]')
    rec_success, rec_failed, img_total, plates_total = txt_recognize(path, ocr)

    print(f'[blink2 yellow]识别图片总数: {img_total}; [blink2 orange]识别车牌总数: {plates_total}; [/blink2 orange]'
          f'[blink2 green]车牌识别成功数: {rec_success}; [/blink2 green][blink2 red]车牌识别失败数{rec_failed}; [/blink2 red]'
          f'[blink2 purple]车牌识别率{round(rec_success * 100 / plates_total, 2)}%[/blink2 purple]')


if __name__ == "__main__":
    console = Console()
    with console.status("[bold yellow]Working on tasks...", spinner='clock') as status:
        with Timer() as timer:
            main()
