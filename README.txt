Convolution.py: Predicts the runtime of convolution as a function of the number of the total FLOPs and Bytes for a given image (total FLOPs = FLOPs per pixel x total number of pixels; Total Bytes = Bytes per pixel times total number of bytes). Refer to your CGMA exercises.
Covariance.py: Predicts runtime of the GPU kernel that performs corner computations as a function of the number of the total FLOPs and Bytes for a given image
convh2d.py and covh2d.py: Predicts host-to-device transfer time as a function of bytes transferred
convd2h.py and covd2h.py: Predicts device-to-host transfer time as a function of bytes transferred
total GPU time prediction = predicted convolution time x number of convolutions + predicted GPU corners time + h2dtimes + d2htimes
