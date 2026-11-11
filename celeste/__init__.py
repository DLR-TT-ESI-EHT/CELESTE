import functools
import numpy as np
import pandas as pd

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python <3.8, fallback to importlib-metadata package
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

# 1. Base - Makes variables for all inputs when creating models

__all__ = ["Base"]


class Base:
    def __init__(self, args):
        for k, v in args.items():  # for each key and value pair
            if k in ['self']:  
                continue
            else:
                self.__dict__[k] = v
                #setattr(self, k, v)

    def rgetattr(self, attr, *args):
        '''
        Get attributes of an object while giving dotted string as input
        '''

        def _getattr(obj, attr):
            if '[' in attr:
                attr_split = attr.split('[')
                idx = int(attr_split[1].replace(']',''))
                return getattr(obj, attr_split[0], *args)[idx]
            else:
                return getattr(obj, attr, *args)
        return functools.reduce(_getattr, [self] + attr.split('.'))

    def rsetattr(self, attr, val):
        '''
        set attributes in object from dotted string path within the object
        '''
        parts = attr.split('.')

        setattr(functools.reduce(getattr, [self] + parts[:-1]), parts[-1], val)

    def to_dict(self):
        """
        Makes dict from models - does not need to be a complete system.
        Can be saved as json - but may fail if it contains numpy floats. 
        If you want to save the model as a json, use the .to_json() method

        """
        res = {}
        for k,v in self.__dict__.items():
            if isinstance(v, Base):  # recursion for other models
                res[f'{k}'] = v.to_dict()
            elif "Medium" in k:  # Only take the repr for Medium model
                res[f'{k}'] = v.__repr__()
            elif k.startswith('_') or k in ["Eq", "args"]:  # Don't save Eq or args
                continue
            else:  # Save everything else
                if isinstance(v, np.ndarray):  #
                    if v.size==1:  # Dealing with single element arrays is annoying
                        res[f'{k}'] = v[0]
                    else:  # Other arrays to list - easy to save as json
                        res[f'{k}'] = v.tolist()
                else:
                    res[f'{k}'] = v
        return res

    def to_pandas(self):
        """
        Converting the dict version of the model to a pandas dataframe with flattened keys.
        The key names may be used to loop through and assign values to the models themselves via setattr() and other python functions
        """
        return pd.json_normalize(self.to_dict(), sep='.')

    def to_json(self, filePath):
        """
        Saves the pandas version of the model as a json. pandas will handle the backend work to change numpy floats to lists etc.

        filePath, str: Path and name of the json file.
        """
        return self.to_pandas().to_json(filePath)

    @staticmethod
    def json_pandas(file):
        df = pd.read_json(file, orient='index')[0]
        return df

    def json_dict(self, file):
        df = self.json_pandas(file)
        return df.to_dict()[0]

