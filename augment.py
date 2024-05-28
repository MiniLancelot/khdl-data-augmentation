from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.2,1.0],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0)


input_dir = 'D:\Avalon\sea_animals_test'

output_dir = "D:\Avalon\sea_animals_augmented"

for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    if os.path.isdir(subdir_path):

        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


        for filename in os.listdir(subdir_path):
            img = load_img(os.path.join(subdir_path, filename))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(output_dir, subdir), save_prefix='aug',
                                      save_format='jpeg'):
                i += 1
                if i > 14:
                    break