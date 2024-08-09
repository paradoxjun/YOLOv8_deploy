class ParsingUrlError(Exception):
    pass


class ReadImageError(Exception):
    pass


class ReadVideoError(Exception):
    pass


class InvalidImageError(Exception):
    pass


class InputFormatError(Exception):
    pass


class PreProcessError(Exception):
    pass


class DetectionInferError(Exception):
    pass


class KeypointPredictionInferError(Exception):
    pass


class PostProcessError(Exception):
    pass
