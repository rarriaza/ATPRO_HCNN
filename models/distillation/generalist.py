from models import ResNetBaseline


class Generalist(ResNetBaseline):
    def __init__(self, n_fine_categories, n_coarse_categories, input_shape, generalist_model_file,
                 logs_directory=None, model_directory=None, args=None):
        super().__init__(n_fine_categories, n_coarse_categories, input_shape,
                         logs_directory, model_directory, args)
        self.load_model(generalist_model_file)
