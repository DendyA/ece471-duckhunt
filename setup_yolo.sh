git clone https://github.com/ultralytics/yolov5.git

cd yolov5

mkdir runs

pip3 install -r requirements.txt

cd ..

cp detect_local.py ./yolov5/detect_local.py

cp -r train ./yolov5/runs/