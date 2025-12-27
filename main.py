import cv2 as cv
import mediapipe as mp
import json
from functions import position_data, calculate_distance, draw_line, overlay_image

def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)

def limit_value(val:int, min_val:int, max_val:int) -> int:
    return max(min(val, max_val), min_val)

def initialize_camera(config: dict) -> cv.VideoCapture:
    cap = cv.VideoCapture(config["camera"]["device_id"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = 12
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # for .mp4
    out = cv.VideoWriter("output.mp4", fourcc, fps, (width, height))
    if not cap.isOpened():
        raise RuntimeError("count not open the camera.")
    return cap, out

def load_images(config: dict) -> tuple:
    inner_circle = cv.imread(config["overlay"]["inner_circle_path"], -1)
    outer_circle = cv.imread(config["overlay"]["outer_circle_path"], -1)
    if inner_circle is None or outer_circle is None:
        raise FileNotFoundError("failed to load images")
    return inner_circle, outer_circle

def process_frame(frame, hands, config, inner_circle, outer_circle, deg):
    h, w, _ = frame.shape
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            wrist, thumb_tip, index_mcp, index_tip, middle_mcp, middle_tip, ring_tip, pinky_tip = position_data(lm_list)
            index_wrist_distance = calculate_distance(wrist, index_mcp)
            index_pinky_distance = calculate_distance(index_tip, pinky_tip)
            ratio = index_pinky_distance / index_wrist_distance

            if 0.5 < ratio < 1.3:
                fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                for finger in fingers:
                    frame = draw_line(frame, wrist, finger,
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])
                for i in range(len(fingers) - 1):
                    frame = draw_line(frame, fingers[i], fingers[i + 1],
                                      color=tuple(config["line_settings"]["color"]),
                                      thickness=config["line_settings"]["thickness"])

            elif ratio >= 1.3:
                center_x, center_y = middle_mcp
                diameter = round(index_wrist_distance * config["overlay"]["shield_size_multiplier"])
                x1 = limit_value(center_x - diameter // 2, 0, w)
                y1 = limit_value(center_y - diameter // 2, 0, h)
                diameter = min(diameter, w - x1, h - y1)
                deg = (deg + config["overlay"]["rotation_degree_increment"]) % 360
                m1 = cv.getRotationMatrix2D((outer_circle.shape[1] // 2, outer_circle.shape[0] // 2), deg, 1)
                m2 = cv.getRotationMatrix2D((inner_circle.shape[1] //2 , inner_circle.shape[0] // 2), -deg, 1)
                rotated_outer = cv.warpAffine(outer_circle, m1, (outer_circle.shape[1], outer_circle.shape[0]))
                rotated_inner = cv.warpAffine(inner_circle, m2, (inner_circle.shape[1], inner_circle.shape[0]))
                frame = overlay_image(rotated_outer, frame, x1, y1, (diameter, diameter))
                frame = overlay_image(rotated_inner, frame, x1, y1, (diameter, diameter))

    return frame, deg

def main():
    config = load_config()
    cap, out = initialize_camera(config)
    inner_circle, outer_circle = load_images(config)
    hands = mp.solutions.hands.Hands()
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    dot_color = (0, 255, 0)
    mesh_color = (0, 255, 0)
    drawing_spec_dots = mp_draw.DrawingSpec(color=dot_color, thickness=1, circle_radius=2)
    drawing_spec_mesh = mp_draw.DrawingSpec(color=mesh_color, thickness=1, circle_radius=1)
    deg = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            faces = face_mesh.process(rgb)
            frame, deg = process_frame(frame, hands, config, inner_circle, outer_circle, deg)
            if faces.multi_face_landmarks:
                for face_landmarks in faces.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION, drawing_spec_mesh, drawing_spec_dots)
            out.write(frame)
            cv.imshow("Image", frame)
            if cv.waitKey(1) == ord(config["keybindings"]["quit_key"]):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
