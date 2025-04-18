import numpy as np
import torch

from . import Flow
from . import CouplingTransform, AffineCouplingLayer, RandomPermutation
from ..distributions   import MultivariateNormalBase

from ...config import CONF_DIR



flow_kwargs = {
    'num_coupling_layers': 5
}

coupling_layers_kwargs = {
    'num_features':     10,
    'num_identity':     5,
    'num_transformed':  5
}

base_distribution_kwargs = {
    'dim':              10,
    'trainable':        True
}

def build_flow( prior_metadata           :dict = None,
                flow_kwargs              :dict = None,
                coupling_layers_kwargs   :dict = None,
                base_distribution_kwargs :dict = None,
                embedding_network_kwargs :dict = None,
                checkpoint_path                = None,
               ):

    
    flow_kwargs              =  kwargs['flow']
    coupling_layers_kwargs   =  kwargs['coupling_layers']
    base_distribution_kwargs =  kwargs['base_distribution']
    embedding_network_kwargs =  kwargs['embedding_network'] 
    configuration = kwargs

    #NEURAL NETWORK ---------------------------------------    
    embedding_network   = Kenn(**embedding_network_kwargs).float()
       
    #BASE DIST ----------------------------------------------------------------------------
    base = MultivariateNormalBase(**base_distribution_kwargs)
    

    #COUPLING TRANSFORM ----------------------------------------------------------------
    coupling_layers = []
    for i in range(flow_kwargs['num_coupling_layers']):
        
        coupling_layers += [RandomPermutation(num_features=coupling_layers_kwargs['num_features'])]

        coupling_layers += [AffineCouplingLayer(coupling_layers_kwargs['num_features'],
                                                embedding_network_kwargs['strain_out_dim'], 
                                                coupling_layers_kwargs['num_identity'], 
                                                coupling_layers_kwargs['num_transformed'])]
        
    coupling_transform = CouplingTransform(coupling_layers)

    #FLOW --------------------------------------------------------------------------------------
    flow = Flow(base_distribution = base, 
                transformation    = coupling_transform, 
                embedding_network = embedding_network, 
                prior_metadata = prior_metadata, 
                configuration = configuration).float()
    
    """loading (eventual) weights"""
    if checkpoint_path is not None:
        flow.load_state_dict(checkpoint['model_state_dict'])
        print('----> Model weights loaded!\n')
        model_parameters = filter(lambda p: p.requires_grad, flow.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'----> Flow has {params/1e6:.1f} M trained parameters')
        
    return flow

