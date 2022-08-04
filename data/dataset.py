import os
from .imagelist import ImageList

# TODO: add PACS dataset

class ltds_dataset(ImageList):
    def __init__(self, root, domain, filter_class=None, split='test', **kwargs):
        if split == 'train':
            self.image_list = {
                "O": "image_list/Original_train.txt",
                "H": "image_list/Hayao_train.txt",
                "S": "image_list/Shinkai_train.txt",
                "V": "image_list/Vangogh_train.txt",
                "U": "image_list/Ukiyoe_train.txt",
            }
        if split == 'test':
            self.image_list = {
                "O": "image_list/Original_test.txt",
                "H": "image_list/Hayao_test.txt",
                "S": "image_list/Shinkai_test.txt",
                "V": "image_list/Vangogh_test.txt",
                "U": "image_list/Ukiyoe_test.txt",
            }
        if split == 'val':
            self.image_list = {
                "O": "image_list/Original_val.txt",
                "H": "image_list/Hayao_val.txt",
                "S": "image_list/Shinkai_val.txt",
                "V": "image_list/Vangogh_val.txt",
                "U": "image_list/Ukiyoe_val.txt",
            }

        self.filelist = os.path.join(root, self.image_list[domain])
        super(ltds_dataset, self).__init__(root, data_list_file=self.filelist, filter_class=filter_class, **kwargs)

if __name__ == '__main__':
    dataset = ltds_dataset(root='/home/dev/Data/xiao/data/awa2-lts/',
                       domain='O',
                       split='train')
    print(dataset)