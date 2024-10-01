import cv2
import numpy as np

def transfer_label(image_path, label_txt_path):
    # label_txt_path include line have the following format:
    # 138.027 590 157.036 580 176.563 570 196.091 560 214.73 550 234.257 540 253.784 530 273.311 520 291.951 510 311.478 500 331.005 490 349.645 480 369.172 470 388.699 460 408.226 450 426.866 440 446.393 430 465.539 420 485.066 410 503.706 400 523.233 390 542.76 380 562.287 370 580.926 360 600.454 350 619.981 340 638.62 330 658.147 320 677.674 310 697.202 300 715.841 290 735.368 280 
    # 1005.75 590 998.474 580 991.42 570 983.778 560 976.724 550 969.669 540 962.027 530 954.973 520 947.331 510 940.277 500 933.223 490 925.581 480 918.527 470 911.473 460 903.831 450 896.777 440 889.723 430 882.473 420 874.831 410 867.777 400 860.135 390 853.081 380 846.026 370 838.385 360 831.33 350 824.276 340 816.634 330 809.58 320 802.526 310 794.884 300 787.83 290 780.776 280 
    # 1667.46 520 1632.31 510 1597.69 500 1562.12 490 1527.5 480 1492.89 470 1457.31 460 1422.7 450 1387.12 440 1352.51 430 1317.89 420 1282.32 410 1247.7 400 1212.75 390 1177.17 380 1142.56 370 1107.94 360 1072.37 350 1037.75 340 1002.18 330 967.561 320 932.947 310 897.371 300 862.757 290 827.181 280 
    
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a blank image with the same dimensions as the original image
    label = np.zeros_like(image)

    # Read the label text file

    with open(label_txt_path, 'r') as file:
        for line in file:
            # Split the line into x and y coordinates
            points = line.strip().split()
            points = list(map(float, points))
            points = list(map(int, points))
            points = np.array(points).reshape((-1, 2))

            # Draw the lane on the blank image
            cv2.polylines(label, [points], isClosed=False, color=(255, 255, 255), thickness=5)

    return image, label

# Example usage:
image_path = 'lane_data/driver_37_30frame/05191310_0427.MP4/00210.jpg'
label_txt_path = 'lane_data/driver_37_30frame/05191310_0427.MP4/00210.lines.txt'
image, label = transfer_label(image_path, label_txt_path)

# Display the original image and the label
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(label)
plt.title('Label')
plt.axis('off')

plt.show()