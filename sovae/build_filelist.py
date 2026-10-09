import argparse
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
    parser = argparse.ArgumentParser(
        description="Recursively write image paths to a text file for SOVAE training."
    )
    parser.add_argument("image_folder", help="Root directory containing images.")
    parser.add_argument("target_file", help="Output text file (one image path per line).")
    args = parser.parse_args()

    collect_images(os.path.abspath(args.image_folder), args.target_file)
