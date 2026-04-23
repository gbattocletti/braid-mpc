class CmdColors:
    """
    Class of ANSI escape sequences to print colored output to the terminal.
    """

    # For more details see:
    # https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal  # pylint: disable=line-too-long
    # ANSI escape sequences color list:
    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
