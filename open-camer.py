import cv2

def main():
    # 打开默认摄像头（通常为0），如果有多个摄像头，请尝试更改此数字
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧（可能是流结束或丢失摄像头连接）")
            break

        # 显示结果帧
        cv2.imshow('摄像头预览', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 完成后释放捕捉器并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()