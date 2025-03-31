import os
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PLM = mp_pose.PoseLandmark

# 오른쪽 측면 랜드마크 목록 (angle 계산 등에서 사용)
RIGHT_LANDMARKS = [
    PLM.RIGHT_SHOULDER,
    PLM.RIGHT_ELBOW,
    PLM.RIGHT_WRIST,
    PLM.RIGHT_INDEX,
    PLM.RIGHT_HIP,
    PLM.RIGHT_KNEE,
    PLM.RIGHT_ANKLE,
    PLM.RIGHT_HEEL,
    PLM.RIGHT_FOOT_INDEX,
]

# 오른쪽 측면 연결 관계
RIGHT_CONNECTIONS = [
    (PLM.RIGHT_SHOULDER, PLM.RIGHT_ELBOW),
    (PLM.RIGHT_ELBOW, PLM.RIGHT_WRIST),
    (PLM.RIGHT_SHOULDER, PLM.RIGHT_HIP),
    (PLM.RIGHT_HIP, PLM.RIGHT_KNEE),
    (PLM.RIGHT_KNEE, PLM.RIGHT_ANKLE),
    (PLM.RIGHT_ANKLE, PLM.RIGHT_HEEL),
    (PLM.RIGHT_HEEL, PLM.RIGHT_FOOT_INDEX),
    (PLM.RIGHT_WRIST, PLM.RIGHT_INDEX),
]

# 이동평균 필터 버퍼 크기
BUFFER_SIZE = 3

# Pose 랜드마크는 0~32까지 총 33개
# 각 랜드마크마다 x, y를 필터링하기 위해 버퍼를 만듭니다.
landmark_buffers = [
    {
        'x': [0.0] * BUFFER_SIZE,  # x좌표 버퍼
        'y': [0.0] * BUFFER_SIZE   # y좌표 버퍼
    }
    for _ in range(33)
]

def mov_avg_filter(buffer_list, new_value):
    """
    buffer_list: 이동평균 버퍼(리스트)
    new_value: 새 측정값
    return: (필터링된 결과값, 갱신된 버퍼)
    """
    buffer_list.pop(0)       # 가장 오래된 값 제거
    buffer_list.append(new_value)  # 새 값 삽입
    return np.mean(buffer_list), buffer_list

def draw_right_side_and_angle(image, landmarks, visibility_th=0.5, angle_offset=15):
    """
    오른쪽 무릎 각도와 손끝-발끝 거리 등을 계산, 화면에 텍스트로 표시
    필터된 랜드마크(landmarks)가 들어온다고 가정.
    """
    h, w, _ = image.shape

    angle_text = ""
    dy_text = ""
    
    hip_lm    = landmarks[PLM.RIGHT_HIP.value]
    knee_lm   = landmarks[PLM.RIGHT_KNEE.value]
    ankle_lm  = landmarks[PLM.RIGHT_ANKLE.value]
    index_lm  = landmarks[PLM.RIGHT_INDEX.value]
    foot_lm   = landmarks[PLM.RIGHT_FOOT_INDEX.value]

    # Check visibility
    if (hip_lm.visibility > visibility_th and 
        knee_lm.visibility > visibility_th and 
        ankle_lm.visibility > visibility_th):

        # 픽셀 좌표 변환 (필터된 좌표임)
        hip_x, hip_y = hip_lm.x * w, hip_lm.y * h
        knee_x, knee_y = knee_lm.x * w, knee_lm.y * h
        ankle_x, ankle_y = ankle_lm.x * w, ankle_lm.y * h
        
        # 무릎 각도
        angle = compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y)
        angle_text = f"Angle: {angle:.0f}°"
        
        if (180 - angle_offset) <= angle <= (180 + angle_offset):
            angle_text += " (Straight)"
            
            # 손끝(Y) - 발끝(Y) 차이 (필터된 좌표로 계산)
            idx_y = index_lm.y * h
            ft_y  = foot_lm.y * h
            dy = (idx_y - ft_y) * 0.092 + 8.5  # 기존 공식 유지
            dy_text = f"hand-feet Y diff: {dy:.2f} cm"

    # 화면에 표시
    if angle_text:
        cv2.putText(image, angle_text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    2, cv2.LINE_AA)
    if dy_text:
        cv2.putText(image, dy_text, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    2, cv2.LINE_AA)

def compute_angle_knee(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
    """
    무릎 각도 계산 (직선일 때 약 180도)
    """
    # Vector from knee to hip
    ux, uy = hip_x - knee_x, hip_y - knee_y
    # Vector from knee to ankle
    vx, vy = ankle_x - knee_x, ankle_y - knee_y
    
    dot = ux * vx + uy * vy
    mag_u = math.sqrt(ux**2 + uy**2)
    mag_v = math.sqrt(vx**2 + vy**2)
    if mag_u * mag_v == 0:
        return 0.0
    
    cos_theta = dot / (mag_u * mag_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def main():
    cap = cv2.VideoCapture(0)
    # 해상도, FPS 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    cap.set(cv2.CAP_PROP_FPS, 27)
    
    os.makedirs('./snapshots', exist_ok=True)
    os.makedirs('./plot', exist_ok=True)
    
    frame_indices = []
    right_index_ys = []
    right_foot_index_ys = []
    
    frame_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            frame_count += 1
            # 프레임 회전(필요한 경우)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # 좌우 반전(셀피 모드)
            frame = cv2.flip(frame, 1)
            
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                h, w = frame.shape[0], frame.shape[1]
                landmarks = results.pose_landmarks.landmark
                
                # 1) 모든 랜드마크(x, y) 필터링
                for i, lm in enumerate(landmarks):
                    # 픽셀 단위 좌표
                    x_px = lm.x * w
                    y_px = lm.y * h

                    # 이동평균 필터 적용
                    filtered_x_px, landmark_buffers[i]['x'] = mov_avg_filter(landmark_buffers[i]['x'], x_px)
                    filtered_y_px, landmark_buffers[i]['y'] = mov_avg_filter(landmark_buffers[i]['y'], y_px)

                    # 다시 normalized 값(0~1 범위)으로 되돌려 저장
                    lm.x = filtered_x_px / w
                    lm.y = filtered_y_px / h
                    # z, visibility는 그대로 사용 (원한다면 z도 비슷하게 필터 가능)

                # 2) 스켈레톤(필터된 좌표) 그리기
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,  
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0),
                        thickness=-1,
                        circle_radius=10
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0),
                        thickness=8
                    )
                )
                
                # 3) 오른쪽 무릎 각도, 손끝-발끝 거리 텍스트
                draw_right_side_and_angle(frame, landmarks)

                # 4) 그래프용 데이터 쌓기 (Right Index와 Right Foot Index의 y값)
                idx_y = landmarks[PLM.RIGHT_INDEX.value].y * h
                ft_y  = landmarks[PLM.RIGHT_FOOT_INDEX.value].y * h
                frame_indices.append(frame_count)
                right_index_ys.append(idx_y)
                right_foot_index_ys.append(ft_y)
            
            # 현재 프레임 저장
            snapshot_path = os.path.join('snapshots', f'frame_{frame_count:04d}.png')
            cv2.imwrite(snapshot_path, frame)

            # 디스플레이
            cv2.imshow('Filtered Skeleton', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC로 종료
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 5) Plot 생성 및 저장
    plt.figure(figsize=(8, 4))
    plt.plot(frame_indices, right_index_ys, label='Filtered Right Index Y')
    plt.xlabel('Frame')
    plt.ylabel('Y position (pixels)')
    plt.title('Filtered Right Index Y vs Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot/right_index_filtered.png')
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.plot(frame_indices, right_foot_index_ys, label='Filtered Right Foot Index Y', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Y position (pixels)')
    plt.title('Filtered Right Foot Index Y vs Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plot/right_foot_index_filtered.png')
    plt.close()
    
    print("Filtered data plots saved to ./plot/")
    print("Screenshots saved to ./snapshots/")

if __name__ == "__main__":
    main()
