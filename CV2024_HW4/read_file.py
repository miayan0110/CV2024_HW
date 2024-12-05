import numpy as np

def read_intrinsics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化變數
    K1, K2 = None, None
    current_camera = None
    matrix_lines = []

    for line in lines:
        line = line.strip()

        # 判斷當前相機名稱
        if "Camera A:" in line:
            current_camera = "K1"
            matrix_lines = []
        elif "Camera B:" in line:
            current_camera = "K2"
            matrix_lines = []

        elif line.startswith("K1:") or line.startswith("K2:"):
            continue

        # 讀取矩陣數值行
        elif current_camera and line:
            matrix_lines.append(line)
            if len(matrix_lines) == 3:
                matrix = []
                for matrix_line in matrix_lines:
                    matrix.append([float(x) for x in matrix_line.split()])
                matrix = np.array(matrix)

                if current_camera == "K1":
                    K1 = matrix
                elif current_camera == "K2":
                    K2 = matrix

    return K1, K2