"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

class MapMode(Enum):
    """
    Enumartion of classifcation modes
    """
    DISTANCE = auto()


@dataclass
class MapConfiguration:
    """
    Represents the configuration for map calculation

    :param reference_embeddings_path: Path to embedded references
    :param volume_embeddings_path: Path to embedded volumes
    :param mode: MapMode to run
    """

    reference_embeddings_path: str
    volume_embeddings_path: str
    output_path: str
    mode: MapMode
    skip_refinement: bool


class MapUI(ABC):
    """Interface to define"""

    @abstractmethod
    def run(self, args=None) -> None:
        """
        Runs the UI.
        :param args: Optional arguments that might need to pass to the parser. Can also be used for testing.
        :return: None
        """

    @abstractmethod
    def get_map_configuration(self) -> MapConfiguration:
        """
        Creates the map configuration and returns it.
        :return: A classification configuration instance
        """
