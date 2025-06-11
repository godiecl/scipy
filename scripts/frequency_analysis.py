#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging

from pydub import AudioSegment

from logger import configure_logging


def main():
    # configure the logger
    configure_logging(logging.DEBUG)

    # get the logger
    log = logging.getLogger(__name__)
    log.debug("Starting main ..")

    file = "..\\data\\songs\\kansas-carry-on-wayward-son.mp3"
    log.debug(f"Reading file: {file} ..")

    audio = AudioSegment.from_file(file)


    log.debug("Done.")


if __name__ == "__main__":
    main()
