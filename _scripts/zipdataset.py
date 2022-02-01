import os
import argparse
import zipfile


# zipfile example
def zip_dir(path):
    zf = zipfile.ZipFile('{}.zip'.format(path), 'w', zipfile.ZIP_DEFLATED)
   
    for root, dirs, files in os.walk(path):
        for file_name in files:
            zf.write(os.path.join(root, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--directory",required=True,help="directory to zip")
    args = parser.parse_args()
    path_to_zip = os.path.dirname(args.directory)
    print("zipping {} to {}.zip".format(path_to_zip,path_to_zip))
    assert os.path.isdir(path_to_zip)
    zip_dir(path_to_zip)
    
    