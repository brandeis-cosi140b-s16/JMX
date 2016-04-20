import os
from shutil import copyfile

copy_path = 'for_IAA_calc'
for dirpath, dirname, filenames in os.walk('Normalized'):
    for filename in filenames:
        if filename.endswith('xml'):
            annotator = dirpath.split('\\')[-2]
            full_path = os.path.join(dirpath, filename)
            base_name, extension = filename.split('.txt')
            new_filename = base_name + '_' + annotator + extension
            new_path = os.path.join(copy_path, new_filename)
            copyfile(full_path, new_path)
