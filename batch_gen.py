import os
import numpy as np
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(31337)


def gen_batch(shape, batch_size):
    image_folder = 'D:/Projects/DeepQuaternionNets/data/segmentation/imgs/'
    mask_folder = 'D:/Projects/DeepQuaternionNets/data/segmentation/masks/'

    # Get list of all images
    image_names = []
    for image_name in os.listdir(image_folder):
        image_names.append(image_name)

    # Augment object
    idg = ImageDataGenerator(rotation_range=0,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=False)

    # Batch Loop
    image_count = len(image_names)
    
    while True:
        X = np.zeros((batch_size, shape[0], shape[1], shape[2]))
        Y = np.zeros((batch_size, 1, shape[1], shape[2]))
        count = 0
        while count < batch_size:
            # Random image and mask
            randi = np.random.randint(0, image_count)
            image_path = os.path.join(image_folder, image_names[randi])
            mask_path = image_path.replace('imgs', 'masks')
            mask_path = mask_path.replace('_', '_road_')

            rs = (shape[1], shape[2], shape[0])
            image = scipy.misc.imread(image_path)
            image = scipy.misc.imresize(image, rs)
            image = image.transpose((2,0,1))
            mask = scipy.misc.imread(mask_path)
            mask = scipy.misc.imresize(mask, rs)
            mask = mask.transpose((2,0,1))
            mask = np.logical_and(mask[0,:,:] == 255, mask[2,:,:] == 255).astype('float32')

            X[count] = (image - 127.5) / 127.5
            Y[count] = mask
            count += 1

        rseed = np.random.randint(0, 1000000)
        xc = 0
        for x in idg.flow(X, batch_size=1, seed=rseed):
            X[xc] = x
            xc += 1
            if xc >= batch_size:
                break
        
        yc = 0
        for y in idg.flow(Y, batch_size=1, seed=rseed):
            Y[yc] = y
            yc += 1
            if yc >= batch_size:
                break

        Y = Y > 0.5

        yield X.astype('float32'), Y.astype('float32')


if __name__ == '__main__':
    bg = gen_batch((3,187,621), 1)
    x,y = next(bg)
    mc = 0