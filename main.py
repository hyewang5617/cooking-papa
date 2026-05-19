import cv2
import sys
from game.game_manager import GameManager


def open_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Camera found at index {idx}")
                return cap
            cap.release()
    return None


def main():
    cap = open_camera()
    if cap is None:
        print("Error: Cannot open webcam. Make sure no other app is using it.")
        sys.exit(1)

    game = GameManager()
    cv2.namedWindow('Vision Cooking Challenge', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        frame = cv2.flip(frame, 1)
        output = game.update(frame)
        cv2.imshow('Vision Cooking Challenge', output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            break
        if key != 0xFF:
            game.handle_key(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
