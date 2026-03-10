import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH="./models/best.pt"
VIDEO_PATH="./2024-10-25_1848.mp4"
YOLO_INPUT_SIZE=416
DESIRED_FPS=3

def resize_with_padding(image,target_size):
    h,w=image.shape[:2]
    scale=min(target_size/w,target_size/h)
    new_w=int(w*scale); new_h=int(h*scale)
    resized=cv2.resize(image,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
    pad_w=(target_size-new_w)//2; pad_h=(target_size-new_h)//2
    padded=cv2.copyMakeBorder(resized,pad_h,target_size-new_h-pad_h,pad_w,target_size-new_w-pad_w,cv2.BORDER_CONSTANT,value=[128,128,128])
    return padded,scale,pad_w,pad_h

def correct_bbox(bbox,scale,pad_w,pad_h,orig_w,orig_h):
    x1=(bbox[0]-pad_w)/scale; y1=(bbox[1]-pad_h)/scale
    x2=(bbox[2]-pad_w)/scale; y2=(bbox[3]-pad_h)/scale
    x1=max(0,min(orig_w,x1)); y1=max(0,min(orig_h,y1))
    x2=max(0,min(orig_w,x2)); y2=max(0,min(orig_h,y2))
    return [int(x1),int(y1),int(x2),int(y2)]

def main():
    detector=YoloDetector(model_path=MODEL_PATH,confidence=0.4)
    tracker=Tracker()
    cap=cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit()

    # Track last known centroid/time per track ID (for speed calculations)
    track_history = {}  # {track_id: (cx, cy, timestamp)}

    frame_delay=int(1000/DESIRED_FPS)
    while True:
        ret,frame=cap.read()
        if not ret: break

        # Ensure the model sees RGB, not BGR (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        orig_h,orig_w=frame.shape[:2]
        resized,scale,pad_w,pad_h=resize_with_padding(frame_rgb,YOLO_INPUT_SIZE)

        detections=detector.detect(resized)

        tracking_ids,boxes=tracker.track(detections,resized)
        now=time.time()

        for tid,bbox in zip(tracking_ids,boxes):
            cb=correct_bbox(bbox,scale,pad_w,pad_h,orig_w,orig_h)
            cx=(cb[0]+cb[2])/2
            cy=(cb[1]+cb[3])/2

            speed=None
            if tid in track_history:
                prev_x,prev_y,prev_t = track_history[tid]
                dt = now - prev_t if now>prev_t else 1e-6
                dx = cx - prev_x
                dy = cy - prev_y
                speed = ((dx*dx + dy*dy)**0.5)/dt
            track_history[tid] = (cx, cy, now)

            # Draw bounding box + ID + speed
            cv2.rectangle(frame,(cb[0],cb[1]),(cb[2],cb[3]),(0,0,255),2)
            cv2.putText(frame,f"ID:{tid}",(cb[0],cb[1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            if speed is not None:
                cv2.putText(frame,f"{speed:.1f}px/s",(cb[0],cb[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)

        cv2.imshow("Frame",frame)
        key=cv2.waitKey(frame_delay)&0xFF
        if key==ord("q") or key==27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
