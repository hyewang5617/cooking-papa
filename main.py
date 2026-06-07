import cv2
import sys
from game import audio
from game.game_manager import GameManager


def open_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS,          60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                print(f"Camera {idx}: "
                      f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
                      f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
                return cap
            cap.release()
    return None


def main():
    cap = open_camera()
    if cap is None:
        print("Error: Cannot open webcam. Make sure no other app is using it.")
        sys.exit(1)

    audio.start_bgm()
    game = GameManager()
    WIN = 'Cooking Papa'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))
        output = game.update(frame)
        cv2.imshow(WIN, output)

        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key != 0xFF:
            game.handle_key(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
