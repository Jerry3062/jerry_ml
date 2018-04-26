import urllib.request
import os
import json

dir_path = 'F:/dataset/iMaterialist Challenge (Furniture) at FGVC5'
images_path = 'F:/dataset/iMaterialist Challenge (Furniture) at FGVC5/validation'
url_file_path = 'validation.json'
url_file_abs_path = os.path.join(dir_path, url_file_path)
with open(url_file_abs_path, 'rb') as test_file:
    test_dic = json.load(test_file)
    images = test_dic['images'][422:]
    annotations = test_dic['annotations']

    not_download = open('F:/dataset/iMaterialist Challenge (Furniture) at FGVC5/validation_not_download.txt', 'a')
    for item, annotation in zip(images, annotations):
        try:
            image_url = item['url'][0]
            label_id = annotation['label_id']
            image_name = str(item['image_id']) + '_' + str(label_id) + '.jpg'
            conn = urllib.request.urlopen(image_url)
            item_image_path = os.path.join(images_path, image_name)
            file_item_image = open(item_image_path, 'wb')
            file_item_image.write(conn.read())
            file_item_image.close()
            print('download ' + image_url)
        except Exception as e:
            not_download.write(image_url + ',' + str(item['image_id']))
            print('not_download ' + image_url)
