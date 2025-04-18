
"""Einstein Telescope detector implementation"""

import os
import glob
from ..config import CONF_DIR
from hyperion.simulations import GWDetector, GWDetectorNetwork


def list_available_detectors():
    #list all the detector files in conf dir
    full_paths = sorted(glob.glob(f"{CONF_DIR}/detectors/*_detector.yml"))
    #extract the detector names
    names = sorted([os.path.basename(path)[:-13] for path in full_paths])
    
    full_paths_dict = {name: path for name, path in zip(names, full_paths)}
    return names, full_paths, full_paths_dict

available_detectors = list_available_detectors()


class EinsteinTelescope(GWDetectorNetwork):
    """
    Wrapper class to Hyperion GWDetectorNetwork class, tuned for Einstein Telescope.
    
    For more details, see hyperion.simulations.GWDetectorNetwork
    """
    
    def __init__(self, **kwargs):
        """
        Constructor
        
        Args:
        -----
            kwargs (dict): optional arguments to pass to the GWDetector constructor
                           See hyperion.simulations.GWDetector for more details
        """
        
        _, _, arms_config_paths = list_available_detectors()
        
        detectors = {}
        for arm in arms_config_paths:
            detectors[arm] = GWDetector(config_file_path=arms_config_paths[arm], 
                                        **kwargs)
        super().__init__(detectors=detectors, **kwargs)
    
    @property
    def arms(self):
        return list(self.detectors.keys())
    
    @property
    def location(self):
        return {arm: self.detectors[arm].location for arm in self.arms}
