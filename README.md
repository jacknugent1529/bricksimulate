# BricksByBits: A generative model for Legos
This repository includes the training for a generative lego model, implemented
with PyTorch and PyTorch Geometric. This model generates lego instructions based
on a voxel "outline."


### Project Setup
1. Make sure [git lfs](https://git-lfs.com/) is installed (run `git lfs install`). This will be used for dataset files.
2. install necessary packages (use `conda env create -f environment.yml`)
3. unzip `data.zip`
4. delete `data/processed` folder
5. Try running `python main.py --data-folder data -B 128 --dim 256 --epochs 10 --num-workers 3 --fast-dev-run`
- will take a while the first time to process the data
- to run a full training cycle (not just a test run), remove the `--fast-dev-run` flag

### Data
The `data.zip` file should contain the following structure:
```
data
├── ldr_files
├── processed
└── raw
```
- ldr_files - this contains ldraw files. These can be viewed with [ldview](https://tcobbs.github.io/ldview/)
- processed - this is generated automatically by the LegoData class when the dataset is loaded
- raw - this contains a pickle file representing the lego model graphs as python dictionaries. For example
    ```python
    {
        'node_labels': [
            'Brick(4, 2)',
            'Brick(2, 4)',
            'Brick(4, 2)',
            ...
            'Brick(4, 2)',
            'Brick(2, 4)'
        ],
        'edges': {
            (0, 1): {'x_shift': -1, 'y_shift': 1},
            (2, 1): {'x_shift': -1, 'y_shift': -1},
            ...
            (9, 11): {'x_shift': 1, 'y_shift': 1},
            (10, 11): {'x_shift': 1, 'y_shift': -1}
        }
    }
    ```

### `LegoModel` Class
We use pytorch geometric `Data` class to hold the Lego model data. 
I also implemented a `LegoModel` class which inherits from `Data`. To use with the dataloader and other utility functions, we should keep all data as `Data` objects, but we can explicitly pass the `self` parameter to use functions from the `LegoModel` class.

For example
```python
    ds = LegoData("../data")
    model = ds[3] # access element from dataset, has type `Data`

    # call the to_voxels LegoModel function on the Data object
    LegoModel.to_voxels(model, 1)
```

