import urllib.request
import os
import json

dir_path = 'F:/dataset/iMaterialist Challenge (Furniture) at FGVC5'
images_path = 'F:/dataset/iMaterialist Challenge (Furniture) at FGVC5/test'
url_file_path = 'test.json'
url_file_abs_path = os.path.join(dir_path, url_file_path)
with open(url_file_abs_path, 'rb') as test_file:
    test_dic = json.load(test_file)
    images = test_dic['images'][5565:]
    not_download = open('F:/dataset/iMaterialist Challenge (Furniture) at FGVC5/test_not_download.txt', 'a')
    for item in images:
        try:
            image_url = item['url'][0]
            image_name = str(item['image_id']) + '.jpg'
            conn = urllib.request.urlopen(image_url)
            item_image_path = os.path.join(images_path, image_name)
            file_item_image = open(item_image_path, 'wb')
            file_item_image.write(conn.read())
            file_item_image.close()
            print('download ' + image_url)
        except:
            not_download.write(image_url + ',' + str(item['image_id']))
            print('not_download ' + image_url)
