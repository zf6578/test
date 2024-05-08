from ultralytics import YOLO
import os 
import shutil
import argparse

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')

    args = parser.parse_known_args()[0] if known else parser.parse_args()

    # 验证输入参数
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    return args

def save_model(src_dir, dst_dir):
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"源目录 {src_dir} 不存在!")
        exit(1)

    try:
        # 遍历源目录及其所有子目录
        for dirpath, dirnames, filenames in os.walk(src_dir):
            # 构造对应的目标目录路径
            rel_path = os.path.relpath(dirpath, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)

            # 如果目标目录不存在，则创建
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            # 拷贝文件
            for filename in filenames:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dst_path, filename)
                shutil.copy2(src_file, dst_file)
    except Exception as e:
        print(f"保存模型过程中发生错误: {e}")
        exit(1)

def main(opt):
    epoch = opt.epochs

    model = YOLO("yolov8n.yaml")
    model = YOLO("/data/pre_model/YOLO.pt")
    results = model.train(data="/root/data/yolo.yaml", epochs=epoch, amp=False, imgsz=1280,batch=50)

    # 保存模型到指定目录
    save_model('./runs/detect/train/', '/data/output/')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


