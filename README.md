# ADL2023

Digging Deeper into Automatic Music Tagging 

This is the coursework submission for COMSM0045 - Applied Deep Learning.

##uncomment to run the basic CNN model with length and stride of 256 for strided convolution
python endtoend.py --model="Basic"

##uncomment to run the basic CNN model with different lengths and strides for the strided convolution
##change the values input into --length-conv and --stride-conv according to the lengths and strides, respectively
#python endtoend.py --model="Basic" --length-conv=1024 --stride-conv=512

##uncomment to run the basic extension to the CNN model- group norm and dropout
#python endtoend.py --epoch=40 --model="Extension1"

##uncomment to run the deep CNN model- extension part 2
##takes 5 hours to run so please have --time around 6 or above
#python endtoend.py --model="Deep"

#sample and annotations in the same folder as the endtoend.py