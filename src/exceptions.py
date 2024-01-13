class ModelNotFoundException(Exception):
    """Raised when the specified model is not found in the available model classes."""
    pass
class InvalidImageSizeException(Exception):
    """Raised when the image size provided is invalid."""
    pass

class InvalidDropoutRateException(Exception):
    """Raised when the dropout rate is not within the valid range (0-1)."""
    pass

class InvalidRegularizerException(Exception):
    """Raised when an invalid regularizer is specified."""
    pass

class InvalidSeedValueException(Exception):
    """Raised when the seed value provided is invalid."""
    pass

class InvalidTrainableLayersException(Exception):
    """Raised when the number of trainable layers is invalid."""
    pass
