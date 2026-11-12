import logging
import sys


from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa: F401

OUTPUT = 25
logging.addLevelName(OUTPUT, "OUTPUT")


def output(self, message, *args, **kwargs):
    if self.isEnabledFor(OUTPUT):
        self._log(OUTPUT, message, args, **kwargs)


logging.Logger.output = output


def configure_logging(logfile=None,
                      loglevel=INFO,
                      logmode="a",
                      outfile=None,
                      output_level=OUTPUT,
                      output_mode="w",
                      console_level=OUTPUT):

    root = logging.getLogger("irrep")

    # remove previous handlers
    root.handlers.clear()

    root.setLevel(loglevel)

    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # console shows plain messages, so OUTPUT lines look the same as the former print() calls
    console_formatter = logging.Formatter(
        "%(message)s"
    )

    output_formatter = logging.Formatter(
        "%(message)s"
    )

    if logfile is not None:
        if isinstance(logfile, str):
            log_handler = logging.FileHandler(logfile, mode=logmode)
        elif hasattr(logfile, "write"):
            log_handler = logging.StreamHandler(logfile)
        elif isinstance(logfile, logging.Handler):
            log_handler = logfile
        else:
            raise ValueError(f"Invalid logfile argument {logfile} of type {type(logfile)}")
        log_handler.setFormatter(log_formatter)
        root.addHandler(log_handler)

    if console_level is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root.addHandler(console_handler)

    if outfile is not None:
        if isinstance(outfile, str):
            output_handler = logging.FileHandler(outfile, mode=output_mode)
        elif hasattr(outfile, "write"):
            output_handler = logging.StreamHandler(outfile)
        elif isinstance(outfile, logging.Handler):
            output_handler = outfile
        else:
            raise ValueError(f"Invalid outfile argument {outfile} of type {type(outfile)}")
        output_handler.setLevel(output_level)
        output_handler.setFormatter(output_formatter)
        root.addHandler(output_handler)

    return root


def set_log_stdout(level=logging.INFO):
    configure_logging(stream=sys.stdout, level=level)


def set_logfile(filename, level=logging.INFO):
    configure_logging(logfilename=filename, level=level)


def set_no_logging():
    root = logging.getLogger("irrep")
    root.handlers.clear()
    root.addHandler(logging.NullHandler())
