"""some utility functions for flow model"""

def get_inference_parameters(inference_parameters_flags, num_overlapping_signals):
    
    inference_parameters = []
    for kind in num_overlapping_signals.keys():
            if num_overlapping_signals[kind] > 0:
                for i in range(num_overlapping_signals[kind]):
                    for par in inference_parameters_flags:
                        inference_parameters.append(f'{par}_{kind}_{i+1}')
                        
    return inference_parameters
