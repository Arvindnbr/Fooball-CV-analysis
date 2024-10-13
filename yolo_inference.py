from ultralytics import YOLO

model = YOLO("/home/arvind/Downloads/best(1).pt")

results = model.predict("/home/arvind/Python/Fooball-CV-analysis/videos/op3.mp4", show = True)
print(results[0])

for det in results:
    print(det.names)


for box in results[0].boxes:
    print(box)
