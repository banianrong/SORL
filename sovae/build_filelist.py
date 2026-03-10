import os

def collect_images(image_folder, target_file):
    cnt = 0
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    with open(target_file, 'w') as f:  
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    f.write(img_path + '\n')
                    cnt += 1
                    
                    if cnt % 10000 == 0:
                        print(f"Current progress: {cnt} images found...", end='\r')

    print(f"\nDone! Included {cnt} images. Saved to {target_file}")

if __name__ == "__main__":
    image_folder = "ILSVRC2012_img_train/" # 文件夹路径，推荐使用绝对路径
    target_file = "xxx.txt" # 输出文件路径

    collect_images(image_folder, target_file)
