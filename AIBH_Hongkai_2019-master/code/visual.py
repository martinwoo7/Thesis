import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SAVEIMG_PATH = '...'


def data2image(num, source_path, file_name):

    csv_file = pd.read_csv(source_path, header = None)
    data = csv_file.as_matrix()
    print("data shape:", data.shape)
    tempdata = data[num].astype(np.uint8)
    tempdata = tempdata[1:]
    temp_r = tempdata[:324]
    temp_g = tempdata[324:648]
    temp_b = tempdata[648:973]
    #temp_r = temp_r.reshape([18,18])
    #temp_g = temp_g.reshape([18,18])
    #temp_b = temp_b.reshape([18,18])
    #temp_rgb = np.stack((temp_r,temp_g, temp_b))
    #print("temp_rgb shape", temp_rgb.shape)


    transform_data = tempdata.reshape([3,18,18])


    #print("temp_r: ", temp_r)

    r = Image.fromarray(transform_data[0],'L').convert('L')
    g = Image.fromarray(transform_data[1],'L').convert('L')
    b = Image.fromarray(transform_data[2],'L').convert('L')

    image = Image.merge('RGB',(r,g,b))
    #image2 = Image.fromarray(temp_rgb)
    #r.save(SAVEIMG_PATH + str(num) + 'r.png','png')
    #g.save(SAVEIMG_PATH + str(num) + 'g.png','png')
    #b.save(SAVEIMG_PATH + str(num) + 'b.png','png')
    #image2.save(SAVEIMG_PATH + str(num) + '_2.png','png')
    image.save(SAVEIMG_PATH + file_name + '.png','png')
    return image

