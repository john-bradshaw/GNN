#!/usr/bin/env python3

"""
Downloads the required files into this directory
"""

import os
from os import path
from urllib import request
import hashlib
import tarfile


def check_file_exists_download_if_not(file_path, link):
    """
    Downloads (if not already exists) a file and checks that it is correct against a hash.
    :param file_path: the path where the downloaded file should live
    :param link: where the file can be downloaded from if it does not exist.
    """

    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            raise RuntimeError("expected file at {}, found folder.".format(file_path))
    else:
        last_percent = [-0.11]
        # ^ lists are imutable so can work with the closure below. saves me having to create a class

        def progress(count, block_size, total_size):
            percent = float(count)*block_size/total_size
            if percent > (last_percent[0] + 0.1):
                last_percent[0] = percent
                print('Downloading {} ({} percent)'.format(file_path, percent))

        filepath, _ = request.urlretrieve(link, file_path, progress)


def incrementally_hash_a_file(fname, hash_creator=None):
    """
    hash a file and return the hexdigest
    :param fname: the filename (including directories) of where the file to hash libs
    :param hash_creator: python function that creates the correct hash. If None then will use md5 hash.
    :return: hexdigest string of file
    """
    # create hash
    if hash_creator is None:
        hasher = hashlib.md5()
    else:
        hasher = hash_creator()

    # go through file and update hash
    with open(fname, "rb") as file_to_hash:
        for chunk in iter(lambda: file_to_hash.read(8192), b""):
            hasher.update(chunk)

    # digest
    digsted = hasher.hexdigest()
    return digsted


def check_file_is_good(filename, expected_hash):
    """
    Checks that the file has the correct hash. If doesn't then will throw an IOError.
    Two main options.
    """
    actual = incrementally_hash_a_file(filename)

    if expected_hash != actual:
        raise IOError("Corrupted file {} as expected hash {} not "
                      "found, actual is {}.".format(filename, expected_hash, actual))
    else:
        return True


def main():
    download_dir = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), 'ro_data')
    # ^ we've actually moved the data outside this file as the QM9 data was making my IDE slow down.
    os.chdir(download_dir)

    print(f"The download directory is: {download_dir}")

    # Download the USPTO data.
    uspto_data_dir = path.join(download_dir, 'uspto')
    os.makedirs(uspto_data_dir, exist_ok=True)

    test_file = path.join(uspto_data_dir, "test.txt")
    check_file_exists_download_if_not(test_file, 'https://www.dropbox.com/s/hzsx9xkowrkfoup/test.txt?raw=1')
    check_file_is_good(test_file, '7ba210be7bc7b08d8a0e2bf74b27e1eb')

    train_file = path.join(uspto_data_dir, "train.txt")
    check_file_exists_download_if_not(train_file, 'https://www.dropbox.com/s/ctu751aifqksi4j/train.txt?raw=1')
    check_file_is_good(train_file, '06fc0c42028b8a487e5f28c1d9093e18')

    valid_file = path.join(uspto_data_dir, "valid.txt")
    check_file_exists_download_if_not(valid_file, 'https://www.dropbox.com/s/bgkxrn6v9kvfrr3/valid.txt?raw=1')
    check_file_is_good(valid_file, '5937f3bd4e9112c1182ad99d0057758d')

    # Download the QM9 data
    qm9_url = 'https://www.dropbox.com/s/8iufk6nyxx7sysj/dsgdb9nsd.xyz.tar.bz2?raw=1'
    qm9_file = path.join(download_dir, 'dsgdb9nsd.xyz.tar.bz2')
    check_file_exists_download_if_not(qm9_file, qm9_url)
    check_file_is_good(qm9_file, "ad1ebd51ee7f5b3a6e32e974e5d54012")
    qm9_output_loc = path.join(download_dir, 'qm9_data')
    if path.isdir(qm9_output_loc):
        print(f"{qm9_output_loc} exists, so not going to re-extract. "
              f"Please manually delete this folder if you wish to rextract")
    else:
        tar = tarfile.open(qm9_file, "r:bz2")
        tar.extractall(qm9_output_loc)  # this can take a while -- a lot of files in there!
        tar.close()
    qm9_file_valid_set = path.join(download_dir, 'qm9_valid.json')
    check_file_exists_download_if_not(qm9_file_valid_set, "https://raw.githubusercontent.com/Microsoft/gated-graph-neural-network-samples/master/valid_idx.json")
    check_file_is_good(qm9_file_valid_set,"0da34f1e9bd25ea5000086f257c00c8c")

    # Download the Zinc Dataset
    zinc_url = 'https://raw.githubusercontent.com/mkusner/grammarVAE/master/data/250k_rndm_zinc_drugs_clean.smi'
    zinc_file = path.join(download_dir, '250k_rndm_zinc_drugs_clean.smi')
    check_file_exists_download_if_not(zinc_file, zinc_url)
    check_file_is_good(zinc_file, '39b977f4dbb35b5e7c694c5676d98363')
    print("DONE!")

if __name__ == '__main__':
    main()
