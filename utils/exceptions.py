class ElementSymbolError(RuntimeError):
    def __init__(self, message="Element not found in periodic table"):
        super().__init__(message)


class NumberOfAtomMismatchError(Exception):
    def __init__(
        self, message="Number of atoms between ligand and ground ligand don't match"
    ):
        super().__init__(message)


class InvalidLigandAfterHsRemoval(Exception):
    def __init__(
        self, message="Forcing the removal of all Hs would result in an invalid ligand"
    ):
        super().__init__(message)


class NoChainsWithinCutoff(Exception):
    def __init__(self, message="No chains found within the given cutoff"):
        super().__init__(message)


class LigandFileLoadingError(Exception):
    def __init__(self, message="No chains found within the given cutoff"):
        super().__init__(message)


class UnkownError(Exception):
    def __init__(self, message="Unkown error"):
        super().__init__(message)
