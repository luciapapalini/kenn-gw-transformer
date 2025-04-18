from optparse import OptionParser


from deepfastet.simulations import WaveformGenerator


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-o", "--output_directory", default='../../gwdata-archive/ET_mdc/training_dataset', 
                                                  help="Output Directory")
    parser.add_option("-k", "--source_kind", default='BBH', help="Source kind: BBH, BNS, NSBH")
    parser.add_option("-n", "--number", default=1000, help="Number of samples to generate")
    parser.add_option("-s", "--save", default=False, action="store_true", help="Save the generated waveforms to disk")

    (options, args) = parser.parse_args()
    
    #read options
    source_kind = options.source_kind
    output_directory = options.output_directory
    N = int(options.number)
    save = options.save

    
    w = WaveformGenerator(source_kind)

    w.run(N, save=save, output_directory=output_directory)
