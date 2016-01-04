def test_convert_o_O(tmpdir):
    import os
    from PIL import Image
    import numpy.testing as nt

    directory = 'tests/ref_data/KDR/sample'
    convert_directory = str(tmpdir)
    crop_size = 512
    extension = 'jpeg'
    n_proc = 4

    os.system(" ".join(["python convert_o_O.py",
                        "--directory", directory,
                        "--convert_directory", convert_directory,
                        "--crop_size", str(crop_size),
                        "--extension", extension,
                        "--n_proc", str(n_proc)]))

    ref_dir = 'tests/ref_data/KDR/sample_o_O_512'

    for filename in os.listdir(ref_dir):
        ref_image = Image.open(os.path.join(ref_dir, filename))
        convert_image = Image.open(os.path.join(convert_directory, filename))
        nt.assert_array_equal(ref_image, convert_image)
