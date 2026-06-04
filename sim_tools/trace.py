"""
Simple functionality aiming to enhanced a users a
ability to trace and debug simulation models.
"""

from abc import ABC
from typing import Optional

from rich.console import Console


DEFAULT_DEBUG = False

CONFIG_ERROR = (
    "Your trace has not been initialised. "
    "Call super__init__(debug=True) in class initialiser"
    "or omit debug for default of no trace."
)


# single rich console - module level.
_console = Console()


# pylint: disable=too-few-public-methods
class Traceable(ABC):
    """
    Provides basic trace functionality for a process to subclass.

    Abstract base class Traceable

    Subclasses must call

    super().__init__(debug=True) in their __init__() method to
    initialise trace.

    Subclasses inherit the following methods:

    trace() - use this function print out a traceable event

    _trace_config(): use this function to return a dict containing
    the trace configuration for the class.  Subclasses should
    override it to implement custom formatting.

    Notes:
    -----
    This class provides the same functionality as the function `trace()`
    in an object orientated framework.  It in theory means cleaner code
    as the call to trace requires less parameters.  However, it must
    be setup correctly.
    """

    def __init__(self, debug: Optional[bool] = DEFAULT_DEBUG):
        """
        Initialise Traceable

        Parameters
        ----------
        debug: bool, Optional (default=False)
            show trace(True). do not show trace (False)
        """
        self.debug = debug
        self._config = Traceable._default_config()

    @classmethod
    def _default_config(cls) -> dict:
        """
        Returns a default trace configuration.
        """
        config = {
            "class": None,
            "class_colour": "bold blue",
            "time_colour": "bold blue",
            "time_dp": 2,
            "message_colour": "black",
            "tracked": None,
        }
        return config

    def _trace_config(self) -> dict:
        """
        Overload to return a custom trace configuration.
        """
        return Traceable._default_config()

    def trace(
        self, time: float, msg: Optional[str] = None, process_id: Optional[str] = None
    ):
        """
        Display a formatted trace of a simulated event.

        Implemented with the rich library Console() object.

        Parameters
        ----------
        time: float
            The simulation time

        msg: str, Optional (default=None)
            Event message to display to user.

        process_id: str, Optional (default=None)
            Display an unique identifer for the trace message .
        """
        # did not initialise trace
        if not hasattr(self, "_config"):
            raise AttributeError(CONFIG_ERROR)

        # if in debug mode
        if self.debug:
            # check for override to default configs
            process_config = self._trace_config()
            self._config.update(process_config)

            # conditional logic to limit tracking to specific processes or
            # entities
            if self._config["tracked"] is None or process_id in self._config["tracked"]:
                # display and format time stamp
                out = (
                    f"[{self._config['time_colour']}]"
                    + f"[{time:.{self._config['time_dp']}f}]:"
                    + f"[/{self._config['time_colour']}]"
                )

                # if provided display and format a process ID
                if self._config["class"] is not None and process_id is not None:
                    out += (
                        f"[{self._config['class_colour']}]"
                        + f"<{self._config['class']} {process_id}>: "
                        + f"[/{self._config['class_colour']}]"
                    )

                # format traced event message
                out += (
                    f"[{self._config['message_colour']}]"
                    + f"{msg}[/{self._config['message_colour']}]"
                )

                # print to rich console
                _console.print(out)


def trace(
    time: float,
    debug: Optional[bool] = DEFAULT_DEBUG,
    msg: Optional[str] = None,
    identifier: Optional[str] = None,
    config: Optional[dict] = None,
):
    """
    Display a formatted trace of a simulated event.

    Implemented with the rich library Console() object.

    Parameters
    ----------
    time: float
        The simulation time.

    debug: bool, Optional (default=False)
        show trace(True). do not show trace (False).

    msg: str, Optional (default=None)
        Event message to display to user.

    identifier: str, Optional (default=None)
        Display an unique identifier for the trace message.

    config: dict, Optional (default=None)
        If None then default colouring is applied to a message.
        Options (with corresponding defaults) include:

            "name":None,
            "name_colour":"bold blue",
            "time_colour":'bold blue',
            "time_dp":2,
            "message_colour":'black',
            "tracked":None

        Use tracked to only show trace for specific process IDs.
        For example if a process are labelled as integers from
        1 to n and we wished to track processes 5, 6 and 25. Then we would set
        tracked = [1, 6, 25]
    """

    # get default and then update with user settings.
    _config = Traceable._default_config()
    if config is None:
        _config["class"] = "event"
    else:
        # update with user settings.
        _config.update(config)

    # if in debug mode
    if debug:
        # conditional logic to limit tracking to specific processes/entities
        if _config["tracked"] is None or identifier in _config["tracked"]:
            # display and format time stamp
            out = (
                f"[{_config['time_colour']}]["
                + f"{time:.{_config['time_dp']}f}]:[/"
                + f"{_config['time_colour']}]"
            )

            # if provided display and format a process ID
            if _config["class"] is not None and identifier is not None:
                out += (
                    f"[{_config['class_colour']}]<{_config['class']} "
                    + f"{identifier}>: [/{_config['class_colour']}]"
                )

            # format traced event message
            out += (
                f"[{_config['message_colour']}]{msg}[/"
                + f"{_config['message_colour']}]"
            )

            # print to rich console
            _console.print(out)
