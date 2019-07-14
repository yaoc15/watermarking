import cv2  
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('./tmp/all_small.jpeg')
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(src.ravel(), 256)
plt.show()
