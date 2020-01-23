
if False:  # used to switch to CPU for testing
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras import optimizers
from digit_cnn import Cnn as digit_cnn
import time
from process_video import *

IMG_DIR = 'graded_images'


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
save_dir = '.'
windowsize = 48
model_type = 'vgg16'
saved_model = 'vgg16_model.h5'
train_it = False  # Set this to True if you want to test a tiny training set
plot_them = False  # Used to plan train, validation, test data

cnn = digit_cnn(model_size=(windowsize, windowsize, 3), save_dir=save_dir)
cnn.build_model(load_saved_model=saved_model, model_type=model_type, opt=sgd)

if train_it:
    subset = .8
    epochs = 2
    cnn.get_data(subset=subset, extra_train=.1, reset=False)
    cnn.train_cnn(epochs=epochs, save_data=None, iter=time.time())

if plot_them:
    data_file_plotter(save_dir, model_type, 'VGG16 Pre-Trained SGD')

# Create an output video.  Set to False unless you want to wait a really long time for
# it to finish.
if False:
    create_video('video.MOV', cnn, windowsize)

# Iterate over all the images in the graded_images folder, run them against the pipeline
# and save them back into the graded_images folder.

scale_values = [(1, 6), (1.25, 24), (1.5, 24)]
for i in range(1, 6):
    start_time = time.time()
    i = str(i)
    print('=' * 50)
    print('   Processing image: ' + i)
    print('=' * 50)
    image = IMG_DIR + '/image' + i + '.png'
    out_image = IMG_DIR + '/' + i + '.png'
    if os.path.isfile(image):
        origin_img = cv2_import_image(image, size=(384, 288))
        img = proj_pipeline(cnn, origin_img, scale_values, windowsize)
        cv2.imwrite(out_image, img)
        print('Image {} total processing time = {}s'.format(i, round(time.time() - start_time), 2))
    else:
        print('File does not exist:', image)

print('')
print('*' * 50)
print('Done!', datetime.now().time())
